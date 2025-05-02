import torch
from torchvision import transforms
from PIL import Image
import os
from lab import ViTSequenceModel  # Make sure this is correct

# Define same preprocessing as during training
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load model from disk
def load_model(model_path, device):
    num_classes = 7  # ← Replace with your actual class count
    sequence_length = 5  # ← Ensure this matches your training
    model = ViTSequenceModel(num_classes=num_classes, sequence_length=sequence_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict from a folder of images
def predict_sequence(folder_path, model=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image file paths and sort
    frame_files = sorted(os.listdir(folder_path))[:5]
    file_paths = [os.path.join(folder_path, f) for f in frame_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not file_paths:
        raise ValueError("No valid images found in the folder.")

    # Load and preprocess images
    images = []
    for file_path in file_paths:
        img = Image.open(file_path).convert('RGB')
        img = image_transforms(img)
        images.append(img)

    # Pad if needed
    while len(images) < 5:
        images.append(torch.zeros_like(images[0]))

    images = torch.stack(images).unsqueeze(0).to(device)  # Shape: [1, 5, 3, 224, 224]

    # Predict
    with torch.no_grad():
        outputs = model(images)
        prediction = torch.argmax(outputs, dim=1).item()

    return prediction
