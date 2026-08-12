import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📱 Mobile Price Classification - ML Models")

# Sidebar for Model Selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

# File uploader
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (test_data.csv)", type=["csv"])

# Get the absolute directory path of where this app.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely join the base directory with the model folder and filenames
model_files = {
    "Logistic Regression": os.path.join(BASE_DIR, "model", "logistic_regression.joblib"),
    "Decision Tree": os.path.join(BASE_DIR, "model", "decision_tree.joblib"),
    "KNN": os.path.join(BASE_DIR, "model", "knn.joblib"),
    "Naive Bayes": os.path.join(BASE_DIR, "model", "naive_bayes.joblib"),
    "Random Forest": os.path.join(BASE_DIR, "model", "random_forest.joblib")
}

if uploaded_file is not None:
    try:
        # Read the test data
        df = pd.read_csv(uploaded_file)
        
        # Ensure 'price_range' is in the dataset
        if 'price_range' not in df.columns:
            st.error("The uploaded CSV must contain the target column 'price_range'.")
        else:
            X_test = df.drop(columns=["price_range"])
            y_test = df["price_range"]
            
            # Load Model
            model_path = model_files[model_choice]
            if not os.path.exists(model_path):
                st.error(f"Model file {model_path} not found. Please run the training script first.")
            else:
                model = joblib.load(model_path)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # Display Metrics
                st.subheader(f"Evaluation Metrics: {model_choice}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")
                
                st.markdown("---")
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload `test_data.csv` from the sidebar to evaluate the models.")