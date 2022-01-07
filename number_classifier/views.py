from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from .models import User, UploadedImage
#from .models import UploadedImage
from PIL import Image


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
        return render(request, "main_page.html")

    elif request.method == "POST":
        print(request.POST)
        print(request.FILES["filename"])
        print(request.FILES)
        image = UploadedImage(image=request.FILES["filename"]).save()

        print("saved image")

        img = UploadedImage.objects().first()
        print(img.image)

        im = Image.open(img.image)
        im.show()
        return render(request, "main_page.html")