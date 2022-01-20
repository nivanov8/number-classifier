import torch
from torch import nn
#from torch.nn.modules.activation import Sigmoid
#from torch.nn.modules.linear import Linear
#from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms.functional as F
import numpy as np
import idx2numpy
import cv2
from PIL import Image


transform_training = transforms.Compose([
    ToTensor(),
    #torchvision.transforms.Grayscale(num_output_channels=1)
    torchvision.transforms.Lambda(lambda x: F.invert(x)),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    #torchvision.transforms.RandomRotation(10)
])

transform_test = transforms.Compose([
    ToTensor(),
    torchvision.transforms.Lambda(lambda x: F.invert(x)),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])


#Download training data from dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_training
)

#Download test data from dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_test
)


    #show image
#print(training_data[4][0].shape)
#plt.imshow(training_data[5][0].permute(1, 2, 0))
#plt.show()

#Create data loaders
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#Get GPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"

#Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(16*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        #print(x.shape)
        x = x.view(x.size(0), -1)

        output = self.out(x)
        return output
        
model = NeuralNetwork().to(device)

#Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #print(X.shape)
        X, y = X.to(device), y.to(device)

        #Compute error
        prediction = model(X)
        loss = loss_function(prediction, y)

        #Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_function(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == "__main__":
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_function, optimizer)
        test(test_dataloader, model, loss_function)
    print("Done!")

    torch.save(model.state_dict(), "number_classifier/Neural_Network/model.pth")
    print("Model saved")
