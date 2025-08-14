# 🧠 Machine Learning Classification Projects

This repository contains **three end-to-end classification projects** that cover **tabular data**, **image data**, and **text data**.  
These projects demonstrate how to preprocess different data types, train models, evaluate performance, and visualize results.

---

## 📂 1. Tabular Data Classification — Heart Disease Prediction

### 📄 Dataset
- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) *(or equivalent)*
- **Goal**: Predict whether a patient has heart disease based on features like age, sex, blood pressure, cholesterol, etc.

### 🔍 Workflow
1. **Data Loading & Cleaning**
   - Handle missing values.
   - Convert categorical features to numerical if needed.
2. **Feature Scaling** to normalize numeric variables.
3. **Model Training**
   - Compared **Random Forest**, **SVM**, **KNN**.
4. **Evaluation**
   - Accuracy score and **confusion matrix** visualizations.

### 📈 Results
| Model         | Accuracy |
|---------------|----------|
| Random Forest | ~0.85    |
| SVM           | ~0.83    |
| KNN           | ~0.81    |

---

## 🖼 2. Image Classification — Handwritten Digits (MNIST)

### 📄 Dataset
- **Source**: Built-in `tensorflow.keras.datasets.mnist`
- **Goal**: Recognize digits (0–9) from 28×28 pixel grayscale images.

### 🔍 Workflow
1. **Preprocessing**
   - Normalize pixel values (0–1).
   - Flatten images for dense layers.
2. **Model Architecture**
   - Dense → ReLU → Dense → Softmax.
3. **Training & Evaluation**
   - Train for 5 epochs.
   - Evaluate accuracy and visualize confusion matrix.

### 📈 Results
- **Accuracy**: ~97%
- **Observation**: Simple dense layers work well; CNNs could improve further.

---

## 🎬 3. Text Classification — IMDB Movie Review Sentiment

### 📄 Dataset
- **Source**: Built-in `tensorflow.keras.datasets.imdb`
- **Goal**: Classify movie reviews as **positive** or **negative**.

### 🔍 Workflow
1. **Preprocessing**
   - Load reviews as word index sequences.
   - Pad sequences to equal length.
2. **Model Architecture**
   - Embedding → GlobalAveragePooling → Dense → Sigmoid.
3. **Training & Evaluation**
   - Train for 10 epochs.
   - Evaluate on test set.

### 📈 Results
- **Accuracy**: ~88%
- **Observation**: Good baseline performance; LSTMs/transformers could do better.

---

## 📚 Key Concepts Learned

### Common Across All Projects
- **Dataset Preparation**: Cleaning, encoding, normalization.
- **Model Training**: Feeding data to adjust weights.
- **Evaluation Metrics**: Accuracy, confusion matrix.
- **Overfitting vs Underfitting**:
  - Overfitting → Model memorizes training data.
  - Underfitting → Model fails to capture patterns.
- **Fine-tuning**: Adjusting hyperparameters & architecture.

### Specific to Each Data Type
- **Tabular** → Feature engineering, categorical encoding, scaling.
- **Image** → Pixel normalization, CNNs for spatial features.
- **Text** → Tokenization, embeddings, sequence padding.

---

## 🚀 Future Improvements

### Tabular
- Try gradient boosting models (XGBoost, LightGBM).
- Use feature selection methods.

### Image
- Implement CNN architectures.
- Use data augmentation.

### Text
- Add LSTM/GRU/Bidirectional RNN layers.
- Use pre-trained embeddings or transformer models (BERT).

---

## 🛠 Technologies Used
- **Languages**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras

---

## 📊 Visuals
Each project includes:
- **Confusion Matrices** for error analysis.
- Accuracy plots for model comparison.
- Data preprocessing steps for reproducibility.

---
## 🖥 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/akshayas02/classification-project.git
cd classification-projects
```

### 2. Install Dependencies
Make sure you have **Python 3.8+** installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Tabular Data Project (Heart Disease Classification)
```bash
cd heart_disease_classification
python main.py
```

### 4. Run the Image Data Project (MNIST Classification)
```bash
cd handwritten_digits
python main.py
```

### 5. Run the Text Data Project (IMDB Sentiment Analysis)
```bash
cd sentiment_analysis
python main.py
```
