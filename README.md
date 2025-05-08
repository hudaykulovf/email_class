# 📧 Business Intent Classifier (BERT + PyTorch)

This project is a real-time business message classifier that uses BERT embeddings and PyTorch to categorize text into intents like:

- 🟥 Complaint  
- 📦 Order Update  
- 📅 Meeting Request  
- 💬 General Feedback  

---

## 🧠 How It Works

```mermaid
graph TD
    A[User Input Message] --> B[BERT Embedding (MiniLM)]
    B --> C[PyTorch Classifier]
    C --> D[Predicted Intent + Confidence]
