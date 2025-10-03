# 📧 Spam Detector with Logistic Regression and Scikit-learn

This project is an implementation of a **Machine Learning model** to classify SMS messages as **spam** or **ham (not spam)**.

---

## 🎯 Project Objective
The main goal is to build a **simple yet effective text classifier**, demonstrating a complete Machine Learning workflow:
- Data loading and preparation
- Text preprocessing
- Model training
- Model evaluation

---

## 📂 Dataset
- **Source:** [Kaggle - Email Spam Detection (mfaisalqureshi)]([https://www.kaggle.com](https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy))
- **Original dataset:** ["SMS Spam Collection" - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,572 English text messages  
- **Labels:** `spam` or `ham`

---

## ⚙️ Methodology
1. **Data Loading**  
   Used **Pandas** to load and explore the dataset (`spam.csv`).

2. **Text Preprocessing**  
   Applied **CountVectorizer** (Scikit-learn) to convert SMS text into numerical vectors based on word frequencies.

3. **Data Splitting**
   - **Training Set (70%)** → Model learns the patterns  
   - **Validation Set (15%)** → Fine-tuning and evaluation during development  
   - **Test Set (15%)** → Final performance measurement  

4. **Model Training**  
   - Used **Logistic Regression** (ideal for binary classification problems).

5. **Evaluation**  
   - Measured **accuracy** on the test set.  

---

## 📊 Results
- **Test Accuracy:** `98.56%`  
- The model effectively differentiates between spam and non-spam messages.  

---

## 💻 Example Code Snippet
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
df.columns = ['label', 'message']

# Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
](https://www.kaggle.com/code/mfaisalqureshi/email-spam-detection-98-accuracy)
