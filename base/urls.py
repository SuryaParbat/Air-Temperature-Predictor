from django.contrib import admin
from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('profile', views.profiling),
    path('predictions', views.predictions),
    path('home', views.home),

]