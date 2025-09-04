# Machine-Learning-Portfolio
Welcome to my Machine Learning Project Portfolio 🚀
This repository showcases practical ML projects covering regression, classification, and predictive modeling, with end-to-end pipelines including data preprocessing, model training, evaluation.

---
## 🎓 Student Performance Predictor
https://github.com/Subith-Varghese/student_performance_predictor

Predict students’ Math scores based on demographic and academic features using ML regression models.

### Key Features
- 📥 Dataset downloaded from Kaggle
- 🔤 Handles categorical (one-hot) & numerical features (scaling)
- 🧑‍💻 Trains multiple regression models: Linear, Ridge, Lasso, Random Forest, XGBoost, CatBoost
- 🌐 Web deployment via Flask for real-time predictions
- 📊 Detailed EDA, visualizations, and feature insights

#### Tech Stack
Python | Pandas | NumPy | Scikit-learn | XGBoost | CatBoost | Matplotlib/Seaborn | Flask

### Workflow
1. Download dataset → data preprocessing → train-test split
2. Model training & evaluation → hyperparameter tuning
3. Save best model & preprocessor → prediction pipeline
4. Deploy via Flask app → user inputs → Math score prediction

#### Example Inputs
- Gender, Race/Ethnicity, Parental Education, Lunch Type, Test Prep, Reading & Writing scores

#### Example Output
- Predicted Math Score: 78 / 100

---
## 💰 Bank Term Deposit Subscription Predictor

Predict whether a client will subscribe to a term deposit based on historical marketing campaign and client data.

### Key Features

- Binary classification with imbalanced dataset (ADASYN oversampling)
- Handles numerical, categorical, and economic indicator features
- Models trained: Logistic Regression, Decision Tree, Random Forest (with hyperparameter tuning)
- Prediction pipeline for unseen customer data
- Optional Flask app for deployment

#### Tech Stack
Python | Pandas | NumPy | Scikit-learn | Imbalanced-learn | Flask

### Workflow
1. Data ingestion → preprocessing → feature selection → train-test split
2. Handle class imbalance → oversampling (ADASYN)
3. Model training & evaluation → save best model & label encoders
4. Prediction pipeline → load new data → preprocess → predict subscription

#### Example Inputs
- Age, Job, Marital status, Education, Default, Housing, Loan, Economic indicators, Campaign info

#### Example Output
- Client subscribes: Yes / No

### Model Performance
| Model               | Accuracy | Precision | Recall | F1-score | Notes                         |
| ------------------- | -------- | --------- | ------ | -------- | ----------------------------- |
| Decision Tree       | 0.96     | 0.79      | 0.82   | 0.80     | Interpretable                 |
| Logistic Regression | 0.83     | 0.37      | 0.82   | 0.52     | Struggles with minority class |
| Random Forest       | 0.96     | 0.85      | 0.82   | 0.84     | Best overall                  |
| Random Forest Best  | 0.96     | 0.83      | 0.83   | 0.83     | Slightly tuned version        |

  

https://github.com/Subith-Varghese/DepositInsight


https://github.com/Subith-Varghese/LDA_csgo-round-winner-prediction

https://github.com/Subith-Varghese/Time-Series-Forecasting-ARIMA-SARIMA/tree/main
