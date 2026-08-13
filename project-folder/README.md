# Machine Learning Assignment 2: Mobile Price Classification

**Live Streamlit App:** https://2025ac05436-kaushik-ai-projects-project-folderapp-9f2bmd.streamlit.app/ 
**GitHub Repository:** https://github.com/2025ac05436-Kaushik/AI_Projects.git 

## a. Problem Statement
The objective is to predict the price range of mobile phones (0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost) based on their hardware specifications such as battery power, RAM, processor cores, and camera features. This is a multi-class classification problem. 

## b. Dataset Description
The model is trained on the "Mobile Price Classification" dataset from Kaggle. 
- **Target Variable:** `price_range` (4 discrete classes)
- **Features:** 20 numerical and categorical features (e.g., `ram`, `battery_power`, `px_width`). 
- **Size:** 2,000 instances, successfully meeting the assignment's minimum criteria of 12 features and 500 instances.

## c. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| **Decision Tree** | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| **KNN** | 0.9350 | 0.9914 | 0.9346 | 0.9350 | 0.9347 | 0.9134 |
| **Naive Bayes** | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| **Random Forest (Ensemble)** | 0.8800 | 0.9769 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |

## d. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest accuracy (96.5%) and AUC (99.87%). This shows that standardizing the data and finding linear decision boundaries works exceptionally well for this specific feature set. |
| **Decision Tree** | Exhibited lower performance (83% accuracy), likely due to overfitting on specific feature thresholds rather than capturing broader patterns in the data. |
| **KNN** | Performed excellently (93.5% accuracy) when used without scaling. This indicates that natural feature scales (like RAM) are heavy and accurate indicators of the target price range. |
| **Naive Bayes** | Provided an acceptable baseline performance (81%). However, its assumption that all features are independent of one another limited its predictive power. |
| **Random Forest (Ensemble)** | A strong ensemble performer (88% accuracy) that overcame single-tree overfitting, though it was surprisingly outperformed by the simpler Logistic Regression model. |
| **Overall Winner for your dataset?** | **Logistic Regression** |