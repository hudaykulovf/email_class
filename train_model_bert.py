import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pickle

# === 1. Load dataset ===
df = pd.read_csv("intents_expanded.csv")

# === 2. Encode labels ===
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# === 3. Device setup for Apple Silicon (MPS) or CPU ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 4. Generate sentence embeddings ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(df['text'].tolist(), convert_to_tensor=True)
X = embeddings.to(device)
y = torch.tensor(labels, dtype=torch.long).to(device)

# === 5. Define classifier model ===
class BERTIntentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BERTIntentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

model = BERTIntentClassifier(input_dim=384, output_dim=len(label_encoder.classes_)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 6. Training loop ===
epochs = 20
for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = loss_fn(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# === 7. Save model and label encoder ===
torch.save(model.state_dict(), "model_bert.pth")
with open("encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ BERT-based model and label encoder saved.")
