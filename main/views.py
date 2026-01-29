import folium
from PIL import Image
from django.shortcuts import render

from .forms import GarbageForm
from .utils import extract_exif_metadata, run_smart_city_pipeline


def index(request):
    center_lat, center_lon = 24.5854, 73.6815
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    m.get_root().width = "800px"
    m.get_root().height = "600px"

    result = None

    if request.method == 'POST':
        form = GarbageForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['img']

            image = Image.open(img_file)
            exif = extract_exif_metadata(image)
            image = image.convert("RGB")

            result = run_smart_city_pipeline(image, image_exif=exif)

            print(result)
    else:
        form = GarbageForm()

    context = {
        'form': form,
        'map': getattr(m.get_root(), '_repr_html_')(),
        'result': result
    }
    return render(request, 'main/index.html', context=context)
