import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FishDataset(Dataset):
    def __init__(self, folder):
        self.imgs = list(Path(folder).glob('*.jpg'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), 0  # dummy label

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28,128), nn.ReLU(),
            nn.Linear(128,2) # dummy 2 classes
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    data_folder = 'output/preprocessed'
    dataset = FishDataset(data_folder)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        for imgs, labels in loader:
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch+1} done')
    torch.save(model.state_dict(), 'output/simple_cnn.pt')
