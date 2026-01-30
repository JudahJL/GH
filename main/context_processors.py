from datetime import datetime, timedelta, timezone
from main.models import TrashTicket


def global_system_status(request):
    """
    Injects real-time system status and stats into every template.
    """
    # 1. Calculate Real Stats
    total = TrashTicket.objects.count()
    open_count = TrashTicket.objects.filter(status=TrashTicket.Status.OPEN).count()
    urgent_count = TrashTicket.objects.filter(severity__gte=7).count()

    # 2. Determine System "Health" Logic
    # Check the last time a ticket was modified/created
    last_active = TrashTicket.objects.order_by('-last_seen').first()

    status_label = "ONLINE"
    status_color = "text-green-500"
    dot_color = "bg-green-500"
    ping_animation = "animate-ping"

    if last_active:
        now = datetime.now(timezone.utc)
        diff = now - last_active.last_seen

        # If activity in last 5 mins -> "RECEIVING DATA" (Blue)
        if diff < timedelta(minutes=5):
            status_label = "RECEIVING DATA"
            status_color = "text-blue-500"
            dot_color = "bg-blue-500"
        # If no activity for 24 hours -> "STANDBY" (Yellow/Amber)
        elif diff > timedelta(hours=24):
            status_label = "STANDBY MODE"
            status_color = "text-amber-500"
            dot_color = "bg-amber-500"
            ping_animation = ""  # Stop pinging if idle

    return {
        # This replaces the manual 'stats' dict in your views
        'stats': {
            'total': total,
            'open': open_count,
            'urgent': urgent_count
        },
        'system_info': {
            'status': status_label,
            'status_class': status_color,
            'dot_class': dot_color,
            'ping': ping_animation,
            'version': 'v2.4.1 (Stable)',
            'node': 'Udaipur Sector-4',
            'last_sync': datetime.now().strftime("%H:%M UTC")
        }
    }