from PIL import Image
from django.contrib import messages
from django.shortcuts import render, redirect
from django.urls import reverse

from .forms import GarbageForm
from .models import TrashTicket
from .utils import extract_exif_metadata, run_smart_city_pipeline, generate_incident_map, generate_heatmap, \
    verify_and_resolve_ticket


def index(request):
    if request.method == 'POST':
        form = GarbageForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['img']

            # Open image for processing
            image = Image.open(img_file)

            # Extract metadata (GPS)
            exif = extract_exif_metadata(image)

            # Ensure RGB for model
            image = image.convert("RGB")

            # Run the AI Pipeline
            result = run_smart_city_pipeline(image, image_exif=exif)
            request.session['result'] = result.pk if type(result) == TrashTicket else result
            return redirect(reverse('index'))
    else:
        form = GarbageForm()

    # --- Fetch Data for Dashboard ---
    tickets = TrashTicket.objects.all().order_by('-first_reported')

    # Calculate simple stats
    total_tickets = tickets.count()
    open_tickets = tickets.filter(status=TrashTicket.Status.OPEN).count()

    # --- Generate Maps ---
    # These functions now return HTML strings (iframe/divs) to embed directly
    incident_map_html = generate_incident_map(tickets)
    heatmap_html = generate_heatmap(tickets)

    result = request.session.pop('result', None)
    if result is not None and type(result) is int:
        result = TrashTicket.objects.get(pk=result)

    context = {
        'form': form,
        'incident_map': incident_map_html,
        'heatmap': heatmap_html,
        'tickets': tickets[:10],
        'result': result,
        'stats': {
            'total': total_tickets,
            'open': open_tickets
        }
    }
    return render(request, 'main/index.html', context=context)


def live_map(request):
    """Renders a full-screen map view."""
    tickets = TrashTicket.objects.all().order_by('-first_reported')

    # Generate maps with 100% height for full-screen experience
    incident_map_html = generate_incident_map(tickets, height="100%")
    heatmap_html = generate_heatmap(tickets, height="100%")

    context = {
        'incident_map': incident_map_html,
        'heatmap': heatmap_html,
        'stats': {
            'total': tickets.count(),
            'open': tickets.filter(status=TrashTicket.Status.OPEN).count()
        }
    }
    return render(request, 'main/live_map.html', context=context)


def ticket_registry(request):
    """Renders the full database registry."""
    tickets = TrashTicket.objects.all().order_by('-first_reported')

    context = {
        'tickets': tickets,
        'stats': {
            'total': tickets.count(),
            'open': tickets.filter(status=TrashTicket.Status.OPEN).count(),
            'urgent': tickets.filter(severity__gte=7).count()
        }
    }
    return render(request, 'main/registry.html', context=context)


def resolve_ticket(request, ticket_id):
    """
    Handles the manual resolution proof upload.
    """
    if request.method == 'POST':
        # 1. Get the uploaded file
        img_file = request.FILES.get('proof_img')

        if not img_file:
            messages.error(request, "Error: No proof image uploaded.")
            return redirect('registry')

        try:
            # 2. Open image (PIL)
            image = Image.open(img_file)

            # 3. Call the Utils Logic
            result = verify_and_resolve_ticket(ticket_id, image)

            # 4. Return Feedback to User
            if result['success']:
                messages.success(request, result['message'])
            else:
                messages.error(request, result['message'])

        except Exception as e:
            print(f"Resolution Error: {e}")
            messages.error(request, "An error occurred while processing the image.")

    return redirect('registry')