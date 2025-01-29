import torch
import torch.nn as nn
import torch.optim as optim
import requests
import os
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse, FileResponse

# ============================
# 1️⃣ Google Drive Model URLs
# ============================
GENERATOR_URL = "https://drive.google.com/uc?id=YOUR_GENERATOR_FILE_ID"
DISCRIMINATOR_URL = "https://drive.google.com/uc?id=YOUR_DISCRIMINATOR_FILE_ID"

# Set model paths
MODEL_DIR = "./models"
GENERATOR_PATH = os.path.join(MODEL_DIR, "generator.pth")
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, "discriminator.pth")

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# 2️⃣ Download Models from Google Drive
# ============================
def download_model(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Model saved at {save_path}")

download_model(GENERATOR_URL, GENERATOR_PATH)
download_model(DISCRIMINATOR_URL, DISCRIMINATOR_PATH)

# ============================
# 3️⃣ Define Models
# ============================
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(features))

# ============================
# 4️⃣ Load Models
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
generator.eval()

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location=device))
discriminator.eval()

# ============================
# 5️⃣ FastAPI App
# ============================
app = FastAPI()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.get("/")
def home():
    return {"message": "Medical Deepfake Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate reconstructed image
    generated_image = generator(image_tensor).squeeze(0).detach().cpu()

    # Check real or fake
    real_or_fake_prob = discriminator(image_tensor).item()
    result = "REAL" if real_or_fake_prob > 0.5 else "FAKE"

    # Convert tensor to image
    generated_pil = transforms.ToPILImage()(generated_image)

    # Save reconstructed image
    output_path = "output.png"
    generated_pil.save(output_path)

    return JSONResponse(content={"prediction": result, "confidence": round(real_or_fake_prob, 2)}, media_type="application/json")

