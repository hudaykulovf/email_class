import torch
import torch.nn as nn
import gradio as gr
import pickle
from sentence_transformers import SentenceTransformer

# === 1. Device setup ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 2. Load encoder ===
with open("encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === 3. Load embedder ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === 4. Rebuild model class ===
class BERTIntentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BERTIntentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# === 5. Load model ===
model = BERTIntentClassifier(input_dim=384, output_dim=len(label_encoder.classes_))
model.load_state_dict(torch.load("model_bert.pth", map_location=device))
model.to(device)
model.eval()

# === 6. Prediction function ===
def predict_intent(text):
    with torch.no_grad():
        embedding = embedder.encode([text], convert_to_tensor=True).to(device)
        output = model(embedding)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        label = label_encoder.inverse_transform([pred_idx.cpu().item()])[0]
        return f"{label} ({confidence.item() * 100:.1f}% confidence)"

# === 7. Gradio UI ===
interface = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=3, placeholder="Enter a business message here..."),
    outputs="text",
    title="📧 Business Intent Classifier",
    description="Classifies messages like complaints, meeting requests, order updates, etc., using BERT embeddings and PyTorch."
)

if __name__ == "__main__":
    interface.launch()
