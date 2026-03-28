import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ------------------ Configuration ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/content/drive/MyDrive/Saved models/kidney_model_final.pth"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

# ------------------ Load Model ------------------
def get_finetune_resnet18(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)
    )
    return model

model = get_finetune_resnet18()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------ Your Transformations ------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Use validation transform for inference
transform = val_transform

# ------------------ Prediction Function ------------------
def predict(image):
    img = Image.fromarray(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    return {CLASSES[i]: float(probs[0][i]) for i in range(len(CLASSES))}

# ------------------ Gradio Interface ------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4),
    title="Kidney CT Scan Classifier",
    description="Upload a kidney CT image to classify as Normal, Cyst, Tumor, or Stone."
)

iface.launch()
