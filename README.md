# number-classifier web application

### Technologies used
- PyTorch
- MongoDB (Mongoengine)
- Django

### Aim
The aim of this project was to get familiar with neural networks as well as practice using a backend framework (Django) and an ORM (mongoengine)

### Challenges
The biggest challenge encountered was implementing the neural network. At first a classic feed forward neural network was used, however the prediction accuracy
was quite low (about 67%). To fix this, a convolutional neural network was implemeted and the prediction accuracy was much better (about 87%).

### The Project
This is a fairly simple project, you can create a new user account, login as an existing user, upload and submit images (hand written numbers). Once the user has submitted the 
image, a prediction of what the number submitted is printed out on the screen. 

### How to run the project
1. Ensure you have PyTorch, Torchvision, mongoengine and django packages installed
2. In the same directory as `manage.py` execute the following command: `python manage.py runserver`
