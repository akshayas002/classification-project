# 🎬 IMDB Movie Review Sentiment Classification

## 📌 Overview
This project demonstrates **text classification** using the **IMDB movie reviews dataset**. The goal is to classify reviews as **positive** or **negative** based on their textual content. We apply machine learning models, evaluate performance, and visualize results.

---

## 📂 Dataset
- **Source:** `tensorflow.keras.datasets.imdb`
- **Type:** Text data (movie reviews)
- **Samples:** 50,000 reviews
- **Features:** Words encoded as integers (top 10,000 most frequent)
- **Target:** Sentiment label (0 = negative, 1 = positive)

---

## 📊 Project Workflow
1. **Load Dataset** – Import from Keras
2. **Preprocessing** – Pad sequences for equal length
3. **Train-Test Split** – Provided 50/50 in dataset
4. **Model Building** – Neural network with:
   - Embedding layer (word vector representation)
   - Global Average Pooling
   - Dense hidden layer
   - Output layer with sigmoid activation
5. **Training** – Binary crossentropy loss, Adam optimizer
6. **Evaluation** – Accuracy score, confusion matrix
7. **Visualization** – Training history plots & heatmaps

---

## 🧠 Model Details
### Architecture:
- **Embedding Layer**: Converts word indices into dense vectors
- **GlobalAveragePooling1D**: Reduces sequence to single vector
- **Dense (ReLU)**: Learns non-linear patterns
- **Dense (Sigmoid)**: Outputs probability for binary classification

---

## 📈 Results

### 🎬 Text Classification (IMDB Movie Reviews)
| Metric     | Value  |
|------------|--------|
| Accuracy   | ~0.88  |

- **Observation:** Good accuracy using a simple embedding + pooling model. Performance could be improved using sequential models like LSTM or transformer-based approaches.

---

## 📚 Key Concepts Learned

- **Embedding Layer**: Converts words into dense numerical vectors.
- **Sequence Padding**: Ensures all input sequences have the same length.
- **Global Average Pooling**: Reduces sequence features while preserving overall meaning.

---

## 🚀 Future Improvements

- Use **LSTM/GRU layers** for sequential pattern recognition.
- Experiment with **Bidirectional RNNs** for better context understanding.
- Incorporate **pre-trained embeddings** like GloVe, Word2Vec, or **transformer-based models** (e.g., BERT).
- Add **dropout** to reduce overfitting.

---

