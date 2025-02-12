from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import io
import uvicorn
from timm.models.vision_transformer import VisionTransformer

app = FastAPI()

# Allowlist the VisionTransformer class
torch.serialization.add_safe_globals([VisionTransformer])

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'model/ViT_classification.pth'
checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = checkpoint['model']
classes = checkpoint['class_names']
model = model.to(device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.post("/predict/")
async def predict(file: UploadFile = File(None)):
    if file is None:
        raise HTTPException(status_code=400, detail="Image not passed")

    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Transform the image
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the class name
    class_name = classes[predicted.item()]

    # Return the prediction
    return JSONResponse(content={"prediction": class_name})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
