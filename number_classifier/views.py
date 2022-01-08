from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from .models import User, UploadedImage
from PIL import Image
from bson import ObjectId
import os
from .Neural_Network.predict import Predict
import base64
from io import BytesIO

# Create your views here.
def index(request):
    if request.method == "GET":
        return render(request, "index.html")

    elif request.method == "POST":
        if "Login" in request.POST:
            return redirect("/main_login")
        elif "Signup" in request.POST:
            return redirect("/main_signup")

def main_login(request):
    if request.method == "GET":
        return render(request, "main_login.html")
    
    elif request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        try:
            user = User.objects().get(username=username, password=password)
            user_id = user.id
            return redirect(f'/main_page/{user_id}')
        
        except User.DoesNotExist:
            return render(request, "main_login.html")

        

def main_signup(request):
    if request.method == "GET":
        return render(request, "main_signup.html")
    
    elif request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        email = request.POST["email"]

        if username == "" or password == "":
            return render(request, "main_signup.html")
        
        elif User.objects(username=username).count() > 0:
            return render(request, "main_signup.html")
        
        else:
            user = User(username=username, password=password, email=email).save() #save the user
            user_id = user.id
            return redirect(f'/main_page/{user_id}')
        

def main_page(request, user_id):
    if request.method == "GET":
        user = User.objects(id=user_id).first()
        images = UploadedImage.objects(user=user).limit(5)

        #test
        '''
        img = UploadedImage.objects().first()
        print(img.image)

        im = Image.open(img.image)
        im.show()'''


        #convert queryset to list
        user_images = []
        for image in images:
            user_images.append(image.image_name)
            #user_images.append(image.image)
            #i.show()
        return render(request, "main_page.html",{"images": user_images})

    elif request.method == "POST":
        if "filename" not in request.FILES:
            return render(request, "main_page.html")
        
        elif "filename" in request.FILES:
            image = request.FILES["filename"]

            #save image to media folder
            im = Image.open(image)
            image_name = str(image)
            path = "C:/Users/nicki/Desktop/number-classifier/media/" + image_name
            im.save(path)

            user = User.objects(id=user_id).first()
            image = UploadedImage(image=image, user=user, path=path, image_name=image_name).save()

            print("saved image")

            '''
            img = UploadedImage.objects().first()
            print(img.image)

            im = Image.open(img.image)
            im.show()
            

            for i in range(2):
                print(os.getcwd())
                os.system("python number_classifier/Neural_Network/predict.py " + )
            '''
            #make a prediction
            predict = Predict(image.image)
            prediction = predict.predict()
            print(prediction)

            
            return redirect(f"/main_page/{user_id}")