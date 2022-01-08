import torch
from torchvision import transforms
from .Model import NeuralNetwork
from PIL import Image
from torchvision.transforms import ToTensor
import sys




class Predict():
    
    class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

    def __init__(self, image):
        self.model = self.reloadModel()
        self.image = self.processImage(image)
        
    
    def reloadModel(self):
        model = NeuralNetwork()
        model.load_state_dict(torch.load(r"C:\Users\nicki\Desktop\number-classifier\number_classifier\Neural_Network\model.pth"))
        return model

    def processImage(self, image):
        convert_tensor = transforms.ToTensor()
        image1 = Image.open(image).convert("L")
        resized_image = image1.resize((28, 28))
        data = convert_tensor(resized_image)
        return data

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.image)
            prediction_index = prediction[0].argmax(0)
            predicted = Predict.class_mapping[prediction_index]
        return predicted