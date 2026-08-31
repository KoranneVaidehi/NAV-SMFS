from django.db import models
from django.contrib.auth.models import User

class MediaUpload(models.Model):
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='media_uploads')
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPES, default='image')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Grad-CAM Heatmap field
    heatmap_file = models.FileField(
        upload_to='heatmaps/%Y/%m/%d/', 
        null=True, 
        blank=True,
        help_text="Grad-CAM heatmap overlay image"
    )
    
    # Prediction details
    prediction_result = models.JSONField(
        null=True, 
        blank=True,
        help_text="Stores prediction and confidence details"
    )
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.file.name} ({self.media_type})"