import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import ViTModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define dataset paths
train_dataset_path = '/Users/chiragagarwal/Desktop/train_sequences/'
val_dataset_path = '/Users/chiragagarwal/Desktop/validation_sequences/'

# Hyperparameters
image_size = 224
sequence_length = 5
batch_size = 8
epochs = 25
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=5):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.data = []
        self.label_map = {}
        self.class_names = []

        # Discover classes and create mapping
        emotion_folders = sorted(os.listdir(root_dir))
        for folder in emotion_folders:
            emotion_path = os.path.join(root_dir, folder)
            if os.path.isdir(emotion_path):
                # Find the next available label
                label = len(self.label_map)
                self.label_map[folder] = label
                self.class_names.append(folder)
                sequences = sorted(os.listdir(emotion_path))
                for sequence_folder in sequences:
                    sequence_path = os.path.join(emotion_path, sequence_folder)
                    if os.path.isdir(sequence_path):
                        self.data.append((sequence_path, label))

        print(f"Discovered {len(self.label_map)} classes:")
        for folder, label in self.label_map.items():
            print(f"{folder} -> {label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_path, label = self.data[idx]
        frames = []
        
        # Load frames
        frame_files = sorted(os.listdir(sequence_path))[:self.sequence_length]
        for frame_file in frame_files:
            frame_path = os.path.join(sequence_path, frame_file)
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Warning: Could not load image {frame_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # Pad sequence if needed
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))
        frames = frames[:self.sequence_length]

        return torch.stack(frames), label

# Load datasets
print("Loading training dataset...")
train_dataset = SequenceDataset(root_dir=train_dataset_path, 
                              transform=train_transforms, 
                              sequence_length=sequence_length)

print("\nLoading validation dataset...")
val_dataset = SequenceDataset(root_dir=val_dataset_path,
                             transform=val_transforms,
                             sequence_length=sequence_length)

# Get actual label information
train_labels = [label for _, label in train_dataset]
val_labels = [label for _, label in val_dataset]

num_classes = len(train_dataset.label_map)
print(f"\nNumber of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training label range: {min(train_labels)} to {max(train_labels)}")
print(f"Validation label range: {min(val_labels)} to {max(val_labels)}")

# Compute class weights
unique_classes = np.unique(train_labels)
print(f"\nUnique labels in dataset: {unique_classes}")
print(f"Label counts: {np.bincount(train_labels)}")

class_weights = compute_class_weight(class_weight='balanced',
                                   classes=unique_classes,
                                   y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

# Model definition
class ViTSequenceModel(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super(ViTSequenceModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit.requires_grad_(False)
        self.sequence_length = sequence_length

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.vit.config.hidden_size,
                nhead=8,
                batch_first=True
            ),
            num_layers=3
        )

        self.fc = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        vit_outputs = self.vit(pixel_values=x).last_hidden_state[:, 0, :]
        vit_outputs = vit_outputs.view(batch_size, seq_len, -1)

        temporal_outputs = self.temporal_transformer(vit_outputs)
        cls_token = temporal_outputs[:, -1, :]
        logits = self.fc(cls_token)
        return logits

# Initialize model
model = ViTSequenceModel(num_classes=num_classes, sequence_length=sequence_length).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training and validation functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
    return epoch_loss / len(loader), epoch_acc / len(loader)

def validate_one_epoch(model, loader, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            epoch_acc += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
    return epoch_loss / len(loader), epoch_acc / len(loader)

# Training loop
best_val_acc = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_vit_sequence_model.pth")
        print("Saved best model!")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()