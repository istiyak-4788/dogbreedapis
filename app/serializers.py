from dataclasses import fields
from rest_framework import serializers
from app.models import Image

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['image']