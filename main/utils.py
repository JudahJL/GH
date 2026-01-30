import io
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional

import cv2
import folium
import numpy as np
import torch
from PIL import Image, ExifTags
from django.conf import settings
from django.core.files.base import ContentFile
from django.db.models import Q
from folium.plugins import HeatMap
from geopy import Nominatim
from google import genai
from skimage.metrics import structural_similarity
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet

from main.models import TrashTicket
from main.pydantic_models import MunicipalReport, DuplicateCheck, ImageExif

CLASS_NAMES = ["0_clean_road", "1_dirty_road", "2_urban_waste"]


def get_torch_model_and_device() -> Tuple[ResNet, torch.device]:
    a_model = models.resnet18(weights='DEFAULT')  # equals IMAGENET1K_V1

    for param in a_model.parameters():
        param.requires_grad = False

    # Find and change the final layers to 3 classes, for resnet-18 it's 512
    num_ftrs = a_model.fc.in_features

    # Replace old one with new one,a nd requires_grad is True by default
    a_model.fc = nn.Linear(in_features=num_ftrs, out_features=3)

    # device agnostic
    a_device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    a_model = a_model.to(a_device)

    # load the pth file to the model (runtime disconnected)
    a_model.load_state_dict(torch.load(settings.MODEL_SAVE_PATH, map_location=a_device))

    return a_model, a_device


model, device = get_torch_model_and_device()

model.eval()


def get_transforms() -> transforms.Compose:
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])


test_transforms = get_transforms()


def dms_to_decimal(dms: Tuple[float, float, float], ref: str) -> float:
    degrees, minutes, seconds = dms
    value = degrees + (minutes / 60) + (seconds / 3600)
    return -value if ref in ('S', 'W') else value


def extract_exif_metadata(img: Image.Image) -> ImageExif:
    exif = img.getexif()

    camera_make = exif.get(ExifTags.Base.Make, None)
    camera_model = exif.get(ExifTags.Base.Model, None)

    gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)

    if not gps_info:
        return ImageExif(camera_make=camera_make, camera_model=camera_model)

    latitude = gps_info.get(ExifTags.GPS.GPSLatitude, None)
    longitude = gps_info.get(ExifTags.GPS.GPSLongitude, None)

    latitude_ref = gps_info.get(ExifTags.GPS.GPSLatitudeRef, None)
    longitude_ref = gps_info.get(ExifTags.GPS.GPSLongitudeRef, None)

    lon, lat = None, None

    if longitude and longitude_ref:
        lon = dms_to_decimal(longitude, longitude_ref)

    if latitude and latitude_ref:
        lat = dms_to_decimal(latitude, latitude_ref)

    return ImageExif(latitude=lat, longitude=lon, camera_make=camera_make, camera_model=camera_model)


def fast_pixel_check(a_img1: Image.Image, a_img2: Image.Image) -> bool:
    img1 = cv2.cvtColor(np.array(a_img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(a_img2), cv2.COLOR_RGB2BGR)

    size = (224, 224,)
    # resize the images to match
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    score, _ = structural_similarity(img1, img2, channel_axis=-1, full=True)
    return score > 0.95  # 95%


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def check_if_duplicate_waste(img_old: Image.Image, img_new: Image.Image):
    prompt_for_duplicate = """
    You are a visual similarity AI for a Smart City.
    Look at Image 1 (Past) and Image 2 (Present).
    Determine if the pile of urban waste in both images is the exact same pile.
    Account for minor changes like wind blowing the trash, lighting changes, or different camera angles.
    """

    # noinspection PyTypeChecker
    response = client.models.generate_content(
        # If server is overloaded use 'gemini-2.5-flash'/'gemini-2.0-flash/'gemini-3-flash-preview'
        model="gemini-2.5-flash-lite",  # Changed model again to an available one
        contents=[prompt_for_duplicate, img_old, img_new],
        config={
            "response_mime_type": "application/json",
            "response_schema": DuplicateCheck
        },
    )

    # report_dict = json.loads(response)
    return response.parsed


def generate_inspector_report(img: Image.Image):
    prompt = """
    You are an automated Municipal Inspector.
    Analyse the street image for urban waste.
    Fill out the JSON report accurately. IF there is no waste, set severity to 0.
    """

    # Send Image and Prompt to gemini
    # noinspection PyTypeChecker
    response = client.models.generate_content(
        # If server is overloaded use 'gemini-2.5-flash'/'gemini-2.0-flash/'gemini-3-flash-preview'
        model="gemini-2.5-flash-lite",  # Changed model again to an available one
        contents=[prompt, img],
        config={
            "response_mime_type": "application/json",
            "response_schema": MunicipalReport
        },
    )

    # report_dict = json.loads(response)
    return response.parsed


def create_or_update_ticket(osm_id: Optional[int] = None, osm_type: Optional[str] = None,
                            img: Optional[Image.Image] = None,
                            ticket: Optional[TrashTicket] = None,
                            image_exif: Optional[ImageExif] = None,
                            gemini_report: Optional[MunicipalReport] = None) -> TrashTicket:
    # manage duplicate and create Municipal ticket
    current_time = datetime.now(timezone.utc)
    buff = io.BytesIO()
    img.save(buff, format='JPEG')

    file = ContentFile(buff.getvalue())
    filename = f"image_{uuid.uuid4()}.jpg"

    if ticket is not None:
        time_elapsed = current_time - ticket.first_reported
        hours_ignored = round(time_elapsed.total_seconds() / 3600, 1)

        # Update the existing tickets
        ticket.last_seen = current_time
        ticket.hours_unattended = hours_ignored
        ticket.image.save(filename, file, save=False)
        ticket.save()

        return ticket

    ticket = TrashTicket.objects.create(osm_type=osm_type, osm_id=osm_id, status=TrashTicket.Status.OPEN,
                                        first_reported=current_time,
                                        last_seen=current_time,
                                        hours_unattended=0.0,
                                        severity=gemini_report.severity_score if gemini_report else 0,
                                        action=gemini_report.action_required if gemini_report else "N/A",
                                        latitude=image_exif.latitude, longitude=image_exif.longitude,
                                        camera_model=image_exif.camera_model)

    ticket.image.save(filename, file, save=False)
    ticket.save()
    return ticket


def run_smart_city_pipeline(img: Image.Image, image_exif: ImageExif):
    if image_exif.latitude is None or image_exif.longitude is None:
        print("Image has no GPS data.")
        return {
            "status": "Error",
            "message": "GPS Missing: This image has no location data. Please upload an original photo taken with location services enabled."
        }

    osm_id, osm_type = None, None

    try:

        nominatim = Nominatim(user_agent="testing")

        location = nominatim.reverse(query=(image_exif.latitude, image_exif.longitude,), exactly_one=True)

        osm_id, osm_type = location.raw['osm_id'], location.raw['osm_type']

    except Exception as e:
        print(f"Geocoding service failed: {e}")

    img_transformed = test_transforms(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_transformed)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred_idx].item()

    ai_label = CLASS_NAMES[pred_idx]
    print(f"Local AI result: {ai_label} ({conf * 100:.1f}% Confidence) ")

    if ai_label == "0_clean_road" and conf > 0.7:
        print("Status: Clean/ Normal. No further action needed.")
        return {"status": "Clean", "action": "None"}

    # The Duplicate Cheks
    is_duplicate = False
    query = Q(osm_id=osm_id) & Q(osm_type=osm_type) & Q(status=TrashTicket.Status.OPEN)
    old_ticket = TrashTicket.objects.filter(query).first()
    if old_ticket:
        old_image = Image.open(old_ticket.image)
        old_image = old_image.convert("RGB")

        # a. Fast pixel check
        if fast_pixel_check(img, old_image):
            print("Same image detected locally")
            is_duplicate = True

        else:
            # b. Gemini Visual Comparison
            print("Running gemini duplication check")
            duplicate_report = check_if_duplicate_waste(old_image, img)
            is_duplicate = duplicate_report.is_same

            if not is_duplicate:
                print("Old trash cleared.")
                old_ticket.status = TrashTicket.Status.RESOLVED
                old_ticket.save(update_fields=['status'])

    # if is_duplicate:
    #     return create_or_update_ticket(img=img, ticket=old_ticket)
    if is_duplicate:
        # Update the ticket in DB
        updated_ticket = create_or_update_ticket(img=img, ticket=old_ticket)

        # --- FIX: Return a Dictionary with "Duplicate" status instead of the object ---
        return {
            "status": "Duplicate",
            "message": "This specific waste pile is already tracked in our system.",
            "severity": updated_ticket.severity,
            "action": updated_ticket.action,
            "hours_unattended": updated_ticket.hours_unattended,
            "ticket_id": updated_ticket.id,
            "waste_type": "Urban Mix"
        }
    else:
        print(" Potential Waste Detected. Requesting for gemini verification...")
        gemini_report = generate_inspector_report(img)

        # check if gemini agrees with model
        if not gemini_report.is_waste_present:
            print("False alarm")
            return {"status": "False Alarm", "action": "None"}

        # generate the ticket
        print("Generating the ticket")
        ticket = create_or_update_ticket(gemini_report=gemini_report,
                                         image_exif=image_exif, osm_id=osm_id, osm_type=osm_type, img=img)
        return ticket


# --- Map Generation Functions (Fixed) ---

def generate_incident_map(tickets, height="350px"):
    """Generates a Folium map with colored markers for tickets."""
    # Centered on Udaipur (or default)
    center_lat, center_lon = 17.3616, 78.4747

    # FIX 1: Set explicit height=350 (pixels) so it doesn't collapse
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, width="100%", height=height)

    # Check if we have tickets to center the map better
    valid_tickets = [t for t in tickets if t.latitude and t.longitude]
    if valid_tickets:
        # Optional: Center map on the latest ticket
        latest = valid_tickets[0]
        m.location = [latest.latitude, latest.longitude]

    for ticket in tickets:
        if ticket.latitude is None or ticket.longitude is None:
            continue

        # Color logic
        if ticket.status == TrashTicket.Status.RESOLVED:
            color = "green"
            status_text = "CLEARED"
        elif ticket.severity >= 7:
            color = "red"
            status_text = "URGENT"
        else:
            color = "orange"
            status_text = "OPEN"

        popup_html = f"""
        <b>Ticket:</b> {ticket.id}<br>
        <b>Status:</b> {status_text}<br>
        <b>Severity:</b> {ticket.severity}/10<br>
        <b>Unattended:</b> {ticket.hours_unattended} hrs
        """

        folium.CircleMarker(
            location=[ticket.latitude, ticket.longitude],
            radius=8 + (ticket.severity / 2),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)

    return m._repr_html_()


def generate_heatmap(tickets, height="350px"):
    """Generates a density heatmap."""
    center_lat, center_lon = 17.3616, 78.4747

    # FIX 2: Removed 'tiles="CartoDB dark_matter"' (Now uses default Light map)
    # FIX 1: Set explicit height=350
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, width="100%", height=height)

    heat_data = []
    for ticket in tickets:
        if ticket.latitude and ticket.longitude:
            # Weight formula
            weight = max(ticket.severity * max(ticket.hours_unattended, 1.0), 1.0)
            heat_data.append([ticket.latitude, ticket.longitude, weight])

    if heat_data:
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            min_opacity=0.4
        ).add_to(m)

    return m._repr_html_()
