import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os

TRAIN_PKL = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/msrvtt_train_rec.pickle"
VAL_PKL = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/msrvtt_val_rec.pickle"
TEST_PKL = "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/msrvtt_test_rec.pickle"


class CorrelationDataset(Dataset):
    def __init__(self, pickle_file):
        self.data = []
        print(f"Loading sequential data from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break
        print(f"Successfully loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matrix, score = self.data[idx]
        return torch.from_numpy(matrix).float(), torch.tensor(score).float()

class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CorrelationDataset(TRAIN_PKL)
val_dataset = CorrelationDataset(VAL_PKL)
test_dataset = CorrelationDataset(TEST_PKL)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

hyperparameters = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "num_epochs": [25],
    "weight_decay": [0, 0.1, 0.01],
}

best_val_loss = float("inf")

for lr in hyperparameters["learning_rate"]:
    for epochs in hyperparameters["num_epochs"]:
        for wd in hyperparameters["weight_decay"]:
            model = CNNRegressor().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            for epoch in tqdm(range(epochs)):
                model.train()
                train_loss = 0
                for images, scores in train_loader:
                    images, scores = images.to(device), scores.to(device)
                    optimizer.zero_grad()
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, scores)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, scores in val_loader:
                        images, scores = images.to(device), scores.to(device)
                        outputs = model(images).squeeze()
                        val_loss += criterion(outputs, scores).item()

                val_loss /= len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/best_model_corr.pth")

model.load_state_dict(torch.load("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/best_model_corr.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to(device))
        predictions.extend(outputs.squeeze().tolist())

with open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/corelation_cnn_datasets/test_predictions.pickle", "wb") as f:
    pickle.dump(predictions, f)