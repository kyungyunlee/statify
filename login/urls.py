from django.urls import path
from .import views
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    path('', views.index, name='index'),
    path('callback/', views.login_callback, name='login_callback'),
    path('main/', views.main, name="main"),
    path('playlists/', views.list_playlists, name="list_playlists"),
    path('playlists/analysis/', views.multiple_analysis, name="multiple_analysis"),
    path('playlists/recommend/', views.compute_recommendation, name="compute_recommendation")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)