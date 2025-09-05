# Machine-Learning-Portfolio
Welcome to my Machine Learning Project Portfolio 🚀
This repository showcases practical ML projects covering regression, classification, and predictive modeling, with end-to-end pipelines including data preprocessing, model training, evaluation.

---
# Regression Projects 
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
# Classification Projects: 
## 💰 Bank Term Deposit Subscription Predictor
https://github.com/Subith-Varghese/DepositInsight
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

---
## 🏆 CSGO Round Winner Prediction 
https://github.com/Subith-Varghese/LDA_csgo-round-winner-prediction

Predict the winning team (CT or T) in CSGO rounds using ML classifiers.

### Key Features:
- LDA-based feature selection → top 20 features
- Models: Logistic Regression, Decision Tree, Random Forest
- Preprocessing: encoding, scaling, feature selection
- Prediction pipeline for unseen match data

### Tech Stack:
Python | Pandas | NumPy | Scikit-learn | LDA | Logging

### Workflow:
1. Preprocess data → handle duplicates → encode categorical features → scale
2. Feature selection via LDA → select top 20 features
3. Train models → save trained models & artifacts
4. Evaluate models → accuracy, classification report, confusion matrix
5. Predict new round winners using saved pipeline

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 75.8%    |
| Decision Tree       | 80.7%    |
| Random Forest       | 85.4% ✅  |

---
# Recommendation System
## 🎬 Netflix Movie Recommendation System
https://github.com/Subith-Varghese/netflix-recommendation-system

Build a personalized movie recommendation system using matrix factorization (SVD).

### Key Features:
- Preprocess Netflix Prize dataset → filter unpopular movies & inactive users
- Train SVD model (matrix factorization) with Surprise library
- Generate Top-N movie recommendations for individual users
- Modular pipeline with separate scripts for preprocessing, training, and recommendation
- Save user-specific recommendations as CSV files

### Workflow:

1. Preprocess ratings dataset → filter by popularity thresholds
2. Train SVD model → save model
3. Generate Top-N recommendations → save CSV per user
4. Logging for pipeline monitoring

---

## Time Series Forecasting

## 🦠 COVID-19 Forecasting Project
https://github.com/Subith-Varghese/covid19-forecasting-analysis

Analyze and forecast COVID-19 trends globally using historical case data.

### Key Features:

- Data cleaning, aggregation, and EDA
- Visualizations: top countries, world maps, bar plots
- Time series forecasting with Facebook Prophet
- Predict future deaths with confidence intervals
- Save forecasts, plots, and trained Prophet model for further use

### Workflow:
1. Load & clean dataset → aggregate country-level data
2. Generate visualizations (Top 10 countries, world maps)
3. Prepare daily deaths for forecasting
4. Train Prophet model → forecast next 14 days
5. Save outputs → plots, forecast CSV, Prophet model
6. 
---

## ✈️ Airline Passenger Forecasting using ARIMA & SARIMA
https://github.com/Subith-Varghese/Time-Series-Forecasting-ARIMA-SARIMA

Forecast international airline passenger traffic using Time Series Analysis.

### Key Features:
- Explore historical passenger trends (Jan 1949 – Dec 1960)
- Check stationarity using ADF Test, Rolling Mean & Standard Deviation
- Apply log transformation & rolling mean differencing to stabilize series
- Build ARIMA and SARIMA models to forecast passenger counts
- Visualize predictions vs actual data
- Forecast future trends for 2 years and 5 years

### Workflow:
1. Load dataset → visualize trends & check stationarity
2. Make series stationary → log transform + rolling differencing
3. Build ARIMA (p,d,q) & SARIMA (p,d,q,s) models
4. Forecast future passenger counts → plot against historical data

### Visualizations:
- Rolling Mean & Std Deviation
- PACF & ACF plots
- ARIMA vs Actual
- SARIMA vs Actual
- 5-Year Forecast

### Concepts Covered:
- Time Series Stationarity & Differencing
- PACF & ACF for ARIMA order selection
- Seasonal ARIMA modeling
- Forecasting future trends
