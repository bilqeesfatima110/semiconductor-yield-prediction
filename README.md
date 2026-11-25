# Semiconductor Yield Prediction
## Overview
This machine-learning project uses sensor measurements from the UCI SECOM dataset to classify manufacturing records as Pass (-1) or Fail (1).

## Dataset
1,567 manufacturing records <br>
590 sensor input features <br>
41,951 missing values <br>
1,463 Pass records and 104 Fail records

## Tools
Python, <br>
Pandas, <br>
Matplotlib, <br>
Scikit-learn, <br>
Imbalanced-learn (SMOTE), and <br>
Joblib.

## Methodology
The target column was separated and the time column excluded from sensor inputs. Data was split into training and testing sets (80%/20%). Missing values were filled with the median, features were scaled using StandardScaler, and PCA retained enough components to explain 95% of training-data variation. SMOTE was applied within the training pipeline. Logistic Regression, Random Forest, and SVM were compared and tuned using GridSearchCV with 3-fold cross-validation focused on the Fail-class F1-score.

## Results
|Model	| Test Accuracy	| Fail F1-score
SVM	| 0.86	| 0.12
Logistic Regression	| 0.81	| 0.12
Random Forest	| 0.92	| 0.07
SVM was selected in this run based on the highest Fail F1-score. Fail detection remained weak, so the model is a baseline and is not production-ready.

## Conclusion
The project demonstrates a classification workflow and highlights the challenge of predicting failures in an imbalanced dataset. Further improvements should focus on increasing Fail recall while limiting false alarms.
