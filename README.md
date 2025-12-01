# Healthcare Insurance Cost Analysis and Prediction

Project Overview

The core objective of this project is to build a robust machine learning model to predict patients' healthcare hospitalization costs and to identify the key medical and demographic factors that contribute most significantly to these predictions. This analysis is crucial for healthcare insurance providers to make strategic and tactical decisions regarding their policies and risk assessment.

Key Analysis & Methodology
Data Integration: Merged three datasets (`Names`, `Medical Examinations`, and `Hospitalisation details`) using the common `Customer ID`.
Data Preprocessing & Feature Engineering:
     Cleaned the data by handling trivial missing values (e.g., removing rows with '?' placeholders).
     Engineered new features, including `age` and `gender` (extracted from the customer name).
     Applied **Ordinal Encoding** to ordered categorical features like `Hospital tier`, `City tier`, and `NumberOfMajorSurgeries`.
Exploratory Data Analysis (EDA) & Statistical Testing: Investigated the impact of various health conditions, lifestyle factors, and location tiers on hospitalization charges using methods like ANOVA, t-tests, and Chi-squared tests.
Machine Learning Modeling: Utilized **Ridge Regression and LightGBM for cost prediction, with model optimization performed using `GridSearchCV` and Stratified K-Fold cross-validation.

Key Findings

Most Influential Factors on Charges (Feature Importance): The primary features driving hospitalization costs are:
    1.  Smoker Status (`smoker_yes`)
    2.  Body Mass Index (BMI)
    3.  Age / Birth Year
    4.  HBA1C (Blood Sugar Level)
Location Impact (ANOVA Test): The statistical analysis confirmed that the average hospitalization costs are significantly different across the three types of City Tiers (Tier 1, Tier 2, Tier 3).

Technical Stack

Language: Python
Core Libraries: `pandas`, `numpy`
Machine Learning: `scikit-learn`, `lightgbm`, `category_encoders`
Statistical Analysis: `scipy.stats`, `statsmodels`
