import torch
import pickle
from tqdm import tqdm

def calculate_pos_weight(dataset):
    labels = [item[1] for item in dataset]
    labels_tensor = torch.tensor(labels)

    num_positives = torch.sum(labels_tensor == 1).item()
    num_negatives = torch.sum(labels_tensor == 0).item()

    if num_positives == 0:
        return torch.tensor([1.0])

    weight = num_negatives / num_positives
    return torch.tensor([weight])

train_dataset = pickle.load(open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_clip_train.pickle", "rb"))
validation_dataset = pickle.load(open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_clip_val.pickle", "rb"))
test_dataset = pickle.load(open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_clip_test.pickle", "rb"))



device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        combined_features, individual_score = self.dataset[index]
        return (
            torch.tensor(combined_features, dtype=torch.float).to(device),
            torch.tensor(individual_score, dtype=torch.float).to(device),
        )

    def __len__(self):
        return len(self.dataset)


train_loader = torch.utils.data.DataLoader(
    CustomDataset(train_dataset), batch_size=256, shuffle=True
)

validation_loader = torch.utils.data.DataLoader(
    CustomDataset(validation_dataset), batch_size=256, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    CustomDataset(test_dataset), batch_size=256, shuffle=False
)


class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

hyperparameters = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "weight_decay": [0, 0.1, 0.01],
}

best_val_loss = float("inf")
best_hyperparams = {}

pos_weight = calculate_pos_weight(train_dataset).to(device)
num_epochs = 25
for lr in hyperparameters["learning_rate"]:
    for decay in hyperparameters["weight_decay"]:
        model = NeuralNetworkClassifier().to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        for epoch in tqdm(range(num_epochs)):
            model.train()
            for combined_features, individual_score in train_loader:

                combined_features = combined_features.squeeze(1)
                pred = model(combined_features).squeeze(1)
                loss = loss_fn(pred, individual_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (
                    combined_features,
                    individual_score,
                ) in validation_loader:
                    combined_features = combined_features.squeeze(1)
                    pred = model(combined_features).squeeze(1)
                    val_loss += loss_fn(pred, individual_score).item()

            val_loss /= len(validation_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = {"learning_rate": lr, "weight_decay": decay}
                print(
                    f"Saving best model with val loss: {best_val_loss}, learing_rate {lr}, weight_decay {decay}"
                )
                torch.save(model.state_dict(), "/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/best_model_msrvtt.pt")
            print("Epoch: ", epoch, "Val loss: ", val_loss, "lr: ", lr, "wd: ", decay)


model = NeuralNetworkClassifier().to(device)
model.load_state_dict(torch.load("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/best_model_msrvtt.pt"))
model.eval()

test_predictions = []
with torch.no_grad():
    for combined_features, individual_score in test_loader:
        combined_features = combined_features.squeeze(1)
        pred = model(combined_features).squeeze(1)
        test_predictions.extend(pred.tolist())
pickle.dump(test_predictions, open("/home/eduard/Desktop/Research/Adrian/VQPP/VAST/clip_datasets/msrvtt_results.pickle", "wb"))