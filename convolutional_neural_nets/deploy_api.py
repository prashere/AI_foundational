from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(
    'models/resnet50_cat_vs_dog.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        if probs.item() > 0.5:
            pred = 1
        else:
            pred = 0

    label = "Dog" if pred == 1 else "Cat"
    return {"prediction": label}
