from . import views
from django.urls import path
urlpatterns = [
    path('', views.index_view, name='index_view'),
    path('file/<str:filename>/', views.file_analysis, name='file_analysis'),
    path('upload/', views.upload_file, name='upload_file'),
    path('dataset/', views.dataset_view, name='dataset_view'),
    path('delete/<str:filename>/', views.delete_file, name='delete_file'),
    path('compare/', views.compare_view, name='compare_view'),
    path('clear-session/', views.clear_session, name='clear_session'), 
]