import os
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import numpy as np
import joblib


BASE_DIR = r"C:\Users\Motasem\Desktop\ML-MOTASEM YILDIZ\archive\Final Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "training")



IMAGE_PATH = r"C:\Users\Motasem\Desktop\ML-MOTASEM YILDIZ\archive\Final Dataset\testing\Traffic Jam\3F12BCA8-1CC1-4726-AF44-3F117A542B9F_cx0_cy9_cw0_w1200_r1.jpg"
dummy_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transforms.ToTensor()
)
CLASS_NAMES = dummy_dataset.classes

print("Classes:", CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


IMG_SIZE_CNN = 160  

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

def create_cnn_model(num_classes: int):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def predict_cnn(image_path: str):
    if not os.path.exists("best_cnn_model.pth"):
        print("ERROR: best_cnn_model.pth not found! Run CNN training first.")
        return None

    model = create_cnn_model(len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = cnn_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]

    print("\n=== CNN Prediction ===")
    print("Predicted class:", pred_class)
    print("Probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        print(f"  {cls:12s}: {p:.3f}")

    return pred_class, probs


IMG_SIZE_RF = 64

rf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_RF, IMG_SIZE_RF)),
    transforms.ToTensor(),
])

def predict_rf(image_path: str):
    if not (os.path.exists("rf_model.joblib") and os.path.exists("rf_scaler.joblib")):
        print("WARNING: Random Forest model or scaler not found. Run RandomForest.py first.")
        return None

    rf = joblib.load("rf_model.joblib")
    scaler = joblib.load("rf_scaler.joblib")

    img = Image.open(image_path).convert("RGB")
    x = rf_transform(img)         
    x = x.view(-1).numpy().reshape(1, -1)  

    x_scaled = scaler.transform(x)

    probs = rf.predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    print("\n=== Random Forest Prediction ===")
    print("Predicted class:", pred_class)
    print("Probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        print(f"  {cls:12s}: {p:.3f}")

    return pred_class, probs


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print("ERROR: IMAGE_PATH does not exist. Please check the path:")
        print(IMAGE_PATH)
    else:
        print(f"\nUsing image: {IMAGE_PATH}")
        predict_cnn(IMAGE_PATH)
        predict_rf(IMAGE_PATH)
