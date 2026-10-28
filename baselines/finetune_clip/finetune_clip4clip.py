import copy
import torch
import pickle
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection

# Same classifier and training procedure as finetune_clip.py, except that the video
# embedding is not the mean of the frame embeddings: the frames are aggregated by the
# temporal transformer of CLIP4Clip (the "seqTransf" similarity header of Luo et al.),
# which is trained together with the classifier.

MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"
NUM_FRAMES = 12  # NUM_FRAMES_TO_SAMPLE in create_dataset_clip4clip.py
NUM_TRANSFORMER_LAYERS = 4  # cross_num_hidden_layers in CLIP4Clip

DATA_DIR = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets"
TRAIN_PATH = f"{DATA_DIR}/vatex_clip4clip_train.pickle"
VAL_PATH = f"{DATA_DIR}/vatex_clip4clip_val.pickle"
TEST_PATH = f"{DATA_DIR}/vatex_clip4clip_test.pickle"
FRAMES_PATH = f"{DATA_DIR}/vatex_clip4clip_frames.pickle"
BEST_MODEL_PATH = f"{DATA_DIR}/best_model_vatex_clip4clip.pt"
RESULTS_PATH = f"{DATA_DIR}/vatex_clip4clip_results.pickle"

def calculate_pos_weight(dataset):
    labels = [item[1] for item in dataset]
    labels_tensor = torch.tensor(labels)

    num_positives = torch.sum(labels_tensor == 1).item()
    num_negatives = torch.sum(labels_tensor == 0).item()

    if num_positives == 0:
        return torch.tensor([1.0])

    weight = num_negatives / num_positives
    return torch.tensor([weight])

train_dataset = pickle.load(open(TRAIN_PATH, "rb"))
validation_dataset = pickle.load(open(VAL_PATH, "rb"))
test_dataset = pickle.load(open(TEST_PATH, "rb"))
frame_embeddings = pickle.load(open(FRAMES_PATH, "rb"))



device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # Samples are ((text embedding, video id), label, query_index, query_text); the
        # query index and text are only needed by the evaluation, so they are dropped here.
        (text_embedding, video_id), individual_score = self.dataset[index][:2]
        return (
            torch.tensor(text_embedding, dtype=torch.float).squeeze(0),
            torch.tensor(frame_embeddings[video_id], dtype=torch.float),
            torch.tensor(individual_score, dtype=torch.float),
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


class TemporalTransformer(torch.nn.Module):
    """The seqTransf header of CLIP4Clip: frame embeddings -> one video embedding.

    As in CLIP4Clip, the frame position embeddings and the transformer layers are
    initialised from the position embeddings and the first layers of the CLIP text
    encoder, the transformer output is added back to the frame embeddings, and the result
    is normalised, mean-pooled over the frames and normalised again.
    """

    def __init__(self, text_model, num_frames, num_layers):
        super(TemporalTransformer, self).__init__()
        clip_text = text_model.text_model
        width = clip_text.config.hidden_size
        if width != text_model.config.projection_dim:
            raise ValueError(
                f"The text transformer width ({width}) differs from the dimension of the "
                f"frame embeddings ({text_model.config.projection_dim})."
            )

        self.frame_position_embeddings = torch.nn.Embedding(num_frames, width)
        self.frame_position_embeddings.weight.data.copy_(
            clip_text.embeddings.position_embedding.weight.data[:num_frames]
        )
        self.layers = torch.nn.ModuleList(
            copy.deepcopy(clip_text.encoder.layers[i]) for i in range(num_layers)
        )

    def forward(self, frames):
        positions = torch.arange(frames.size(1), device=frames.device)
        hidden = frames + self.frame_position_embeddings(positions)
        for layer in self.layers:
            # No attention mask: every video has the same number of frames. No causal
            # mask either: each frame attends to all the others.
            hidden = layer(hidden, None, None)[0]
        hidden = hidden + frames

        hidden = hidden / hidden.norm(dim=-1, keepdim=True)
        video = hidden.mean(dim=1)
        return video / video.norm(dim=-1, keepdim=True)


class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self, text_model):
        super(NeuralNetworkClassifier, self).__init__()
        self.temporal_transformer = TemporalTransformer(
            text_model, NUM_FRAMES, NUM_TRANSFORMER_LAYERS
        )
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, text_embedding, frames):
        video_embedding = self.temporal_transformer(frames)
        x = torch.cat((text_embedding, video_embedding), dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Only used to initialise the temporal transformer of every model.
clip_text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME)

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
        model = NeuralNetworkClassifier(clip_text_model).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        for epoch in tqdm(range(num_epochs)):
            model.train()
            for text_embedding, frames, individual_score in train_loader:
                text_embedding, frames, individual_score = (
                    text_embedding.to(device), frames.to(device), individual_score.to(device)
                )
                pred = model(text_embedding, frames).squeeze(1)
                loss = loss_fn(pred, individual_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for text_embedding, frames, individual_score in validation_loader:
                    text_embedding, frames, individual_score = (
                        text_embedding.to(device), frames.to(device), individual_score.to(device)
                    )
                    pred = model(text_embedding, frames).squeeze(1)
                    val_loss += loss_fn(pred, individual_score).item()

            val_loss /= len(validation_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = {"learning_rate": lr, "weight_decay": decay}
                print(
                    f"Saving best model with val loss: {best_val_loss}, learing_rate {lr}, weight_decay {decay}"
                )
                torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Epoch: ", epoch, "Val loss: ", val_loss, "lr: ", lr, "wd: ", decay)


model = NeuralNetworkClassifier(clip_text_model).to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

test_predictions = []
with torch.no_grad():
    for text_embedding, frames, individual_score in test_loader:
        pred = model(text_embedding.to(device), frames.to(device)).squeeze(1)
        # The model is trained with BCEWithLogitsLoss, so it outputs logits. Store
        # probabilities instead, because the evaluation thresholds these values.
        test_predictions.extend(torch.sigmoid(pred).tolist())
pickle.dump(test_predictions, open(RESULTS_PATH, "wb"))
