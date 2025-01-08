from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
urlpatterns = [
    path('', views.index_view, name='index_view'),
    path('file/<str:filename>/', views.file_analysis, name='file_analysis'),
    path('distribution', views.distribution_view, name='distribution_view'), 
    path('word_count', views.word_view, name='word_view')]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)