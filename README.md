# 💬 Sentiment Analysis of Product Reviews

Classify customer reviews as **Positive**, **Negative**, or **Neutral** using traditional ML (SVM + TF-IDF) and deep learning (BERT/DistilBERT).

---

## 👥 Team Members & Contributions

| Name | Branch | Role |
|------|--------|------|
| Adhithya | `feature/eda` | EDA, visualizations, category-wise analysis |
| Keerthana | `feature/model` | Preprocessing, TF-IDF, SVM, BERT, model saving |
| Fidal | `feature/deployment` | Streamlit UI, Flask API, README, deployment |

---

## 📁 Project Structure

```
sentiment-analysis-product-reviews/
│
├── app.py                  # Streamlit web app (UI)
├── flask_app.py            # Flask REST API
├── model.ipynb             # Model training notebook
├── svm_model.pkl           # Trained SVM model
├── tfidf.pkl               # TF-IDF vectorizer
├── Reviews.csv             # Dataset
├── requirements.txt        # Python dependencies
├── Procfile                # For deployment (Heroku/Render)
└── test_api.py             # API test script
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Keerthanajp18/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

### 4. Run the Flask API (optional)
```bash
python flask_app.py
```
Then test it:
```bash
python test_api.py
```

---

## 🔗 API Usage (Flask)

**Endpoint:** `POST /predict`

**Request Body (JSON):**
```json
{
  "review": "This product is amazing and works perfectly!"
}
```

**Response:**
```json
{
  "review": "This product is amazing and works perfectly!",
  "sentiment": "positive"
}
```

---

## 🧠 Models Used

| Model | Features | Method |
|-------|----------|--------|
| SVM | TF-IDF vectors | Traditional ML |
| DistilBERT | Contextual embeddings | Deep Learning |

---

## 📊 Dataset

- **Source:** Amazon Product Reviews
- **Labels:** Positive / Negative / Neutral
- **File:** `Reviews.csv`

---

## 🛠 Tech Stack

- Python, Scikit-learn, NLTK
- HuggingFace Transformers (BERT)
- Flask (REST API)
- Streamlit (Web UI)

- stramlit link
-  https://sentiment-analysis-appuct-reviews-bxc7cbhtkz4mvrut84y2yt.streamlit.app/

- streamlit screenshot
- Screenshot 2026-05-16 145533.png
