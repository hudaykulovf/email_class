import torch
import torch.nn as nn
import gradio as gr
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# === 1. Load model architecture ===
class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# === 2. Load vectorizer and label encoder ===
with open("vectorizer.pkl", "rb") as f:
    vectorizer, label_encoder = pickle.load(f)

input_dim = len(vectorizer.get_feature_names_out())
hidden_dim = 128
output_dim = len(label_encoder.classes_)

model = IntentClassifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# === 3. Prediction function ===
def classify_intent(text):
    x = vectorizer.transform([text]).toarray()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = torch.softmax(output, dim=1).max().item()
    return f"{label} ({confidence*100:.1f}% confidence)"

# === 4. Gradio UI ===
interface = gr.Interface(
    fn=classify_intent,
    inputs=gr.Textbox(lines=3, placeholder="Enter email message here..."),
    outputs="text",
    title="Business Email Intent Classifier",
    description="Predicts the intent of short business messages (e.g., complaints, meeting requests, etc.)"
)

if __name__ == "__main__":
    interface.launch()
