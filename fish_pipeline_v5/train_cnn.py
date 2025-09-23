import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import json
from pathlib import Path

# Load config
with open('config.json') as f:
    config = json.load(f)

PREPROCESSED_DIR = os.path.join(config['paths']['output'], 'preprocessed')
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']

class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.label_map = {}
        for i, species in enumerate(os.listdir(root_dir)):
            species_dir = os.path.join(root_dir, species)
            if not os.path.isdir(species_dir): continue
            self.label_map[i] = species
            for img_file in os.listdir(species_dir):
                self.samples.append(os.path.join(species_dir, img_file))
                self.labels.append(i)
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (TARGET_SIZE//8) * (TARGET_SIZE//8), 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((config['preprocessing']['target_size'], config['preprocessing']['target_size'])),
        transforms.ToTensor()
    ])
    dataset = FishDataset(PREPROCESSED_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.label_map)
    model = SimpleCNN(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} completed.")
    torch.save(model.state_dict(), os.path.join(config['paths']['output'], 'cnn_model.pt'))
    print("Model saved.")
