
# 📧 Business Intent Classifier (BERT + PyTorch)

This project is a real-time business message classifier that uses BERT embeddings and PyTorch to categorize text into intents like:

- 🟥 Complaint
- 📦 Order Update
- 📅 Meeting Request
- 💬 General Feedback

---

## 🧠 How It Works

```mermaid
graph TD;
    A[User Input Message] --> B[BERT Embedding (MiniLM)]
    B --> C[PyTorch Classifier]
    C --> D[Predicted Intent + Confidence]
```

- **Embeddings**: Sentence-Transformer `all-MiniLM-L6-v2`
- **Classifier**: Simple `nn.Linear` model trained with `CrossEntropyLoss`
- **Interface**: Built with [Gradio](https://gradio.app/) for fast testing and deployment

---

## 📦 File Structure

```
email-intent-pytorch/
├── intents_expanded.csv       # Training data (200+ labeled samples)
├── train_model_bert.py        # Training script
├── app_bert.py                # Gradio app interface
├── model_bert.pth             # Trained PyTorch model
├── encoder.pkl                # Label encoder (sklearn)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model_bert.py
```

3. Launch the app:
```bash
python app_bert.py
```

Then go to [http://localhost:7860](http://localhost:7860)

---

## ✨ Features Coming Soon

- ✅ Augmented dataset for improved accuracy
- ✅ Multi-label classification support
- ✅ Explainable predictions (keyword-based clues or confidence chart)
- ✅ Prediction logging (for monitoring & retraining)
- ✅ Deployment on Hugging Face Spaces

---

## 🧠 Example Inputs

| Message                           | Predicted Label  |
|----------------------------------|------------------|
| I would like to refund my order. | Complaint        |
| Can we meet tomorrow?            | Meeting Request  |
| Where is my shipment?            | Order Update     |
| Thanks for your support!         | General Feedback |

---

## 🛠 Tech Stack

- `sentence-transformers`
- `PyTorch`
- `scikit-learn`
- `Gradio`

---

## 🤝 Author

Built with ❤️ by Farrukh using M4 Mac and way too much debugging.
