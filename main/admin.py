from django.contrib import admin

from main.models import TrashReport, TrashTicket

# Register your models here.
for model in (TrashReport, TrashTicket):
    admin.site.register(model)
