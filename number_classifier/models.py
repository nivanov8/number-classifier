from typing_extensions import Required
from django.db import models
from mongoengine import Document, StringField, ImageField, ReferenceField
from PIL import Image

# Create your models here.
class User(Document):
    username = StringField(max_length=20, required=True)
    password = StringField(max_length=20, required=True)
    email = StringField(max_length=30, required=False)

class UploadedImage(Document):
    image = ImageField()
    user = ReferenceField(User, required=True)
    path = StringField()
    image_name = StringField()
