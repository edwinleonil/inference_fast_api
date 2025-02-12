# load a pytorch model from: model/ResNet50.pth
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import io

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'model/ResNet50.pth'
checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = checkpoint['model']
classes = checkpoint['class_names']
print('claseses are:', classes)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Read the image file
image_path = "C:/Users/me1elar/Documents/GitHub/AI-23-19-AVI-TransferLearning/data/200525_7classes/test/Pinsite/X,Y,Z2022-04-05 09_46_44.667_Z.png"

image = Image.open(image_path).convert("RGB")

# Transform the image
image = transform(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Get the class name
class_name = classes[predicted.item()]

# Return the prediction
print(class_name)
