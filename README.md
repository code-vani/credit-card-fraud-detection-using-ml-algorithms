# 💳 Credit Card Fraud Detection Using Machine Learning  

## 📝 Problem Statement  
Credit card fraud has become one of the biggest challenges in the financial sector. With millions of daily transactions, identifying fraudulent activities is crucial to prevent financial losses.  

The main problem is that fraudulent transactions are very rare (~0.17%) compared to normal transactions, making the dataset highly imbalanced. Traditional machine learning models often fail to detect such rare events, hence the need for specialized approaches to handle imbalance and build robust fraud detection systems.  

---

## 🎯 Objective  
The goal of this project is to:  
- Detect fraudulent credit card transactions using machine learning models.  
- Handle imbalanced data using undersampling techniques.  
- Compare multiple models and identify the best-performing classifier.  

---

## 📊 Dataset  
We use the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) from Kaggle.  

- **Total Transactions**: 284,807  
- **Fraudulent Transactions**: 492 (~0.17%)  
- **Normal Transactions**: 284,315 (~99.83%)  
- **Observation**: Dataset is highly imbalanced, which makes fraud detection challenging.  

---

## ⚙️ Workflow  
1. **Import Libraries** → pandas, numpy, matplotlib, seaborn, sklearn.  
2. **Data Exploration (EDA)**  
   - Checked for missing values (none found).  
   - Visualized class distribution (high imbalance).  
   - Correlation analysis using heatmap.  
3. **Data Balancing**  
   - Used **undersampling** to balance fraud and normal transactions.  
4. **Data Splitting**  
   - 80% training, 20% testing.  
5. **Model Training & Hyperparameter Tuning**  
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Decision Tree  
   - Random Forest  
   - Gaussian Naive Bayes  
   - K-Nearest Neighbors (KNN)  
   - Used **GridSearchCV** for hyperparameter tuning.  
6. **Model Evaluation**  
   - Compared models using **accuracy** and **best parameters**.  

---

## 📈 Results  

| Model                | Accuracy | Best Parameters                           |  
|-----------------------|----------|-------------------------------------------|  
| Logistic Regression   | 92.9%    | {'solver': 'liblinear'}                   |  
| SVM                  | 91.7%    | {'kernel': 'rbf', 'C': 10}                |  
| Decision Tree         | 89.7%    | {'criterion': 'entropy'}                  |  
| Random Forest         | **93.8%** | {'criterion': 'entropy', 'n_estimators': 50} |  
| Gaussian Naive Bayes  | 86.7%    | Default                                   |  
| KNN                  | 63.7%    | {'n_neighbors': 5}                        |  

✅ **Best Model → Random Forest (93.8% accuracy)**  

---

## 🚀 Conclusion  
- The **Random Forest Classifier** gave the **highest accuracy (93.8%)** among all models.  
- Fraudulent transactions are very rare, hence data balancing is crucial.  
- Future improvements can be achieved by:  
  - Using **advanced resampling techniques** (SMOTE, ADASYN).  
  - Applying **deep learning models** like LSTM/Autoencoders.  
  - Exploring **anomaly detection approaches**.  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Libraries**: pandas, numpy, matplotlib, seaborn, sklearn  

---

## ▶️ How to Run  

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the notebook/script

bash
Copy code
jupyter notebook credit_card_fraud_detection.ipynb
or

bash
Copy code
python fraud_detection.py
