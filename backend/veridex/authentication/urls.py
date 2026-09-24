from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path("", views.dashboard_view, name="home"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard_view, name="dashboard"),

    # Upload
    path("upload/", views.upload_page, name="upload_page"),
    path("upload/submit/", views.upload_media, name="upload_media"),

    # Media
    path("delete/<int:media_id>/", views.delete_media, name="delete_media"),
    path("api/media/<int:media_id>/", views.get_media_info, name="get_media_info"),
]