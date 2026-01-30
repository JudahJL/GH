from django.urls import path

from main import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live-map/', views.live_map, name='live_map'),
    path('registry/', views.ticket_registry, name='registry'),
]
