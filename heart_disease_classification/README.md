# 🫀 Heart Disease Classification  
A machine learning project that predicts the presence of heart disease using tabular data. This project compares multiple classification models (Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)) and evaluates them with visual metrics like confusion matrices.  

## 📌 Features  
- Data preprocessing & cleaning  
- Train-test split  
- Multiple model training & comparison  
- Evaluation with accuracy, classification report, and confusion matrix  
- Visualization of model performance  

## 📂 Dataset  
- **Name:** Heart Disease Dataset  
- **Source:** [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)  
- **Format:** CSV  
- **Target column:** `target` (1 = disease present, 0 = no disease)  

## ⚙️ Installation  
1. Clone the repository  
   `git clone https://github.com/<your-username>/heart-disease-classification.git`  
   `cd heart-disease-classification`  
2. Install dependencies  
   Make sure you’re using Python 3.10+ and run:  
   `pip install -r requirements.txt`  

**requirements.txt**  
pandas
numpy
matplotlib
scikit-learn
seaborn

markdown
Copy
Edit

## 🚀 Usage  
Run the script:  
`python heart_disease_classification.py`  

## 📊 Workflow  
1. **Data Loading:** Load the CSV file into a pandas DataFrame.  
2. **Data Preprocessing:** Separate features (X) and target (y), split into training and test sets (80%-20%).  
3. **Model Training:** Train three different models – Random Forest, SVM, and KNN.  
4. **Evaluation:** Calculate accuracy, generate classification report, plot confusion matrices, and compare performance visually.  

## 📈 Results  
| Model         | Accuracy |  
|---------------|----------|  
| Random Forest | 0.85     |  
| SVM           | 0.83     |  
| KNN           | 0.78     |  

Confusion matrices were plotted for each model to provide deeper insight into correct and incorrect predictions.  

## 🧠 Concepts Covered  
- **Classification basics:** Predicting categories from features  
- **Model fitting & training:** Feeding the algorithm with training data so it learns patterns  
- **Fine-tuning:** Adjusting hyperparameters to improve performance  
- **Overfitting vs Underfitting:**  
  - Overfitting → model learns training data too well and performs poorly on unseen data  
  - Underfitting → model fails to learn enough from training data and performs poorly everywhere  
- **Performance evaluation metrics:** Accuracy, precision, recall, F1-score, confusion matrix  

## 🛠️ Future Improvements  
- Add cross-validation  
- Implement hyperparameter tuning (GridSearchCV)  
- Deploy as a web app with Flask or Streamlit  
