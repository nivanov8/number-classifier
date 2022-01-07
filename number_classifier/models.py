from django.db import models
from mongoengine import Document, StringField
from mongoengine.fields import ImageField
from PIL import Image

# Create your models here.
class User(Document):
    username = StringField(max_length=20, required=True)
    password = StringField(max_length=20, required=True)
    email = StringField(max_length=30, required=False)

class UploadedImage(Document):
    image = ImageField(upload_to="images/")
