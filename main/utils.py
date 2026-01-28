import uuid
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from django.conf import settings
from geopy import Nominatim
from google import genai
from skimage.metrics import structural_similarity
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet

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


def fast_pixel_check(a_img1: Image.Image, img2_path: str) -> bool:
    print("DEBUG old_image_path type:", type(img2_path), img2_path)

    img1 = cv2.cvtColor(np.array(a_img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    size = (224, 224,)
    # resize the images to match
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    score, _ = structural_similarity(img1, img2, full=True)
    return score > 0.95  # 95%


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def check_if_duplicate_waste(old_image_path: str, img_new: Image.Image):
    img_old = Image.open(old_image_path)

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


TICKET_DB = {}


def create_or_update_ticket(location_id: str, image_path: str, is_duplicate=False, gemini_report=None, latitude=None,
                            longitude=None, camera_model=None):
    # manage duplicate and create Municipal ticket
    current_time = datetime.now()

    if is_duplicate:
        existing_ticket = TICKET_DB[location_id]
        time_elapsed = current_time - existing_ticket["first_reported"]
        hours_ignored = round(time_elapsed.total_seconds() / 3600, 1)

        # Update the existing tickets
        existing_ticket['last_seen'] = current_time
        existing_ticket['hours_unattended'] = hours_ignored
        existing_ticket['latest_image_path'] = image_path

        return existing_ticket

    ticket_id = f"TKT-{str(uuid.uuid4())[:4].upper()}"

    new_ticket = {
        "ticket_id": ticket_id,
        "location_id": location_id,
        "status": "OPEN",
        "first_reported": current_time,
        "last_seen": current_time,
        "hours_unattended": 0.0,
        "severity": gemini_report.severity_score if gemini_report else 0,
        "action": gemini_report.action_required if gemini_report else "N/A",
        "latest_image_path": image_path,
        "latitude": latitude,
        "longitude": longitude,
        "camera_model": camera_model,
    }

    TICKET_DB[ticket_id] = new_ticket
    print(f"Ticket created: {ticket_id} for {location_id}")
    return new_ticket


def run_smart_city_pipeline(img: Image.Image, image_exif: ImageExif, location_id="Ward_5_Lake_Road"):
    nominatim = Nominatim(user_agent="testing")
    pincode: str = nominatim.reverse((image_exif.latitude, image_exif.longitude,), exactly_one=True).raw['address']['postcode']
    # TODO: put django filepath from django file storage
    image_path = ""
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

    if location_id in TICKET_DB and TICKET_DB[location_id]['status'] == 'OPEN':
        old_image_path = TICKET_DB[location_id]['latest_image_path']

        # a. Fast pixel check
        if fast_pixel_check(img, old_image_path):
            print("Same image detected locally")
            is_duplicate = True

        else:
            # b. Gemini Visual Comparison
            print("Running gemini duplication check")
            duplicate_report = check_if_duplicate_waste(old_image_path, img)
            is_duplicate = duplicate_report.is_same

            if not is_duplicate:
                print("Old trash cleared.")
                TICKET_DB[location_id]['status'] = "RESOLVED"

    if is_duplicate:
        return create_or_update_ticket(location_id, image_path, is_duplicate=True)
    else:
        print(" Potential Waste Detected. Requesting for gemini verification...")
        gemini_report = generate_inspector_report(img)

        # check if gemini agrees with model
        if not gemini_report.is_waste_present:
            print("False alarm")
            return {"status": "False Alarm", "action": "None"}

        # generate the ticket
        print("Generating the ticket")
        ticket = create_or_update_ticket(location_id, image_path, is_duplicate=False, gemini_report=gemini_report,
                                         latitude=image_exif.latitude, longitude=image_exif.longitude,
                                         camera_model=image_exif.camera_model)
        return ticket
