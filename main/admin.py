from django.contrib import admin

from main.models import TrashTicket

# Register your models here.
for model in (TrashTicket,):
    admin.site.register(model)
