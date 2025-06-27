import torch
from torchvision import transforms, models
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(
    'models/resnet50_cat_vs_dog.pth', map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Cat", "Dog"]

def predict(img):
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

    cat_confidence = 1 - probs  
    dog_confidence = probs     

    return {"Cat": float(cat_confidence), "Dog": float(dog_confidence)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier",
    description="Upload an image of a cat or a dog and see the prediction!"
).launch()
