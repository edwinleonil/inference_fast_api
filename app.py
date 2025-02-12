import logging
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security.api_key import APIKeyHeader
from typing import List
import torch
from torchvision import models, transforms
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

API_KEY = "your-secret-api-key"
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )


# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'model/ResNet50.pth'
checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model = checkpoint['model']
classes = checkpoint['class_names']
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.post("/predict", dependencies=[Depends(get_api_key)])
async def predict(image: UploadFile = File(...)):
    logger.info("Received prediction request")
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_name = classes[predicted.item()]
    logger.info(f"Prediction result: {class_name}")
    return {"class_name": class_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# test the API from a different PC with the following command:
curl - X POST "http://172.16.49.197:8000/predict" - H "accept: application/json" - H "access_token: your-secret-api-key" - F "image=@C:\Users\me1elar\Documents\pin.png"
