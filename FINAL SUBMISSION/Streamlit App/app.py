
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt

# Define available models
model_options = {
    "Baseline DNN (Best on Validation Set)": "best_dnn_model.keras",
    "Fairness-Aware DNN (Reweighted)": "reweighted_model.keras",
    "Final DNN (Last Epoch Model)": "final_dnn_model.keras"
}

# Load preprocessing tools
pipeline = joblib.load("pipeline.pkl")
encoder = joblib.load("encoder.pkl")

# Streamlit layout
st.set_page_config(page_title="Fair Credit Score Predictor", layout="wide")
st.title("Fair Credit Score Prediction App")
st.markdown("This application predicts an individual's credit score category using different trained machine learning models.")

# Sidebar with model info
st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
model = load_model(model_options[selected_model_name])
st.sidebar.markdown("""Each model was trained using different strategies to balance accuracy and fairness.

- **Baseline DNN**: Best accuracy on validation set  
- **Fairness-Aware**: Trained using reweighing to reduce bias  
- **Final DNN**: Last epoch snapshot  
""")

# Input form layout using columns
st.header("Enter Applicant Information")
with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income", min_value=1000.0, value=45000.0)
        salary = st.number_input("Monthly In-hand Salary", value=3500.0)
        num_bank_accounts = st.slider("Number of Bank Accounts", 0, 20, value=2)
        num_credit_cards = st.slider("Number of Credit Cards", 0, 15, value=3)
        interest_rate = st.slider("Interest Rate (%)", 0, 40, value=15)
    with col2:
        num_loans = st.slider("Number of Loans", 0, 10, value=1)
        num_inquiries = st.slider("Number of Credit Inquiries", 0, 20, value=2)
        delay_days = st.slider("Delay from Due Date (days)", 0, 30, value=5)
        num_delayed = st.slider("Number of Delayed Payments", 0, 15, value=1)
        changed_limit = st.number_input("Change in Credit Limit", value=1000.0)
        credit_util_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.5, value=0.6)
    with col3:
        credit_history = st.slider("Credit History Age (months)", 0.0, 400.0, value=150.0)
        monthly_investment = st.number_input("Amount Invested Monthly", value=300.0)
        payment_min = st.selectbox("Payment of Minimum Amount", ["No", "Yes"])
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        spending = st.selectbox("Spending Habit", ["Low", "High"])
        payment = st.selectbox("Payment Size", ["Small", "Medium", "Large"])
        occupation = st.selectbox("Occupation", ["Scientist", "Engineer", "Teacher", "Doctor", "Artist", "Other"])

    submitted = st.form_submit_button("Predict Credit Score")

if submitted:
    input_data = pd.DataFrame([{
        'Age': age,
        'Annual_Income': income,
        'Monthly_Inhand_Salary': salary,
        'Num_Bank_Accounts': num_bank_accounts,
        'Num_Credit_Card': num_credit_cards,
        'Interest_Rate': interest_rate,
        'Num_of_Loan': num_loans,
        'Num_Credit_Inquiries': num_inquiries,
        'Delay_from_due_date': delay_days,
        'Num_of_Delayed_Payment': num_delayed,
        'Changed_Credit_Limit': changed_limit,
        'Outstanding_Debt': income * credit_util_ratio,
        'Credit_Utilization_Ratio': credit_util_ratio,
        'Credit_History_Age': credit_history,
        'Amount_invested_monthly': monthly_investment,
        'Payment_of_Min_Amount': payment_min,
        'Credit_Mix': credit_mix,
        'Spending_Habit': spending,
        'Payment_Size': payment,
        'Occupation': occupation
    }])

    X_input = pipeline.transform(input_data)

    # Debug info for diagnosis
    st.subheader("Debug Info")
    st.write("Transformed Input Shape:", X_input.shape)
    st.write("Raw Prediction Probabilities:", model.predict(X_input))

    pred_probs = model.predict(X_input)[0]
    pred_class = np.argmax(pred_probs)
    credit_label = encoder.inverse_transform([pred_class])[0]

    # Summary box
    st.subheader("Prediction Summary")
    st.success(f"Predicted Credit Score Category: **{credit_label}**")
    st.markdown(f"This profile is most likely categorized as **{credit_label}**, based on the selected model: *{selected_model_name}*.")

    # Bar chart for probabilities
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        "Class": encoder.classes_,
        "Probability": pred_probs
    }).sort_values(by="Probability", ascending=False)

    st.bar_chart(prob_df.set_index("Class"))
