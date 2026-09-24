from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from authentication import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # React / Django application
    path('', include('authentication.urls')),

    # React API
    path('api/analyze/', views.upload_media, name='api_analyze'),
]

urlpatterns += static(
    settings.MEDIA_URL,
    document_root=settings.MEDIA_ROOT
)