from PIL import Image
from geopy import Nominatim

from main.utils import extract_exif_metadata

img = Image.open("assets/20260126_130130.jpg")
exif = extract_exif_metadata(img)
a = Nominatim(user_agent="testing")
b = a.reverse(query=(exif.latitude, exif.longitude,), exactly_one=True, addressdetails=True)
print(b.raw)

from datetime import date
from main.models import TrashTicket

ticket = TrashTicket.objects.create(
    osm_id=12345678,
    osm_type="node",  # could be 'node', 'way', 'relation'
    status=TrashTicket.Status.OPEN,
    first_reported=date.today(),
    last_seen=date.today(),
    severity=7,  # 0–10
    action="Needs municipal pickup",
    image="trash_imgs/sample.jpg",  # path relative to MEDIA_ROOT
    latitude=40.7128,
    longitude=-74.0060,
    camera_model="iPhone 15 Pro"
)

print(ticket.id)
