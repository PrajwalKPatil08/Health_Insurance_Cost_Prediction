import streamlit as st
import json
import hashlib
import re
import pandas as pd
import os
from datetime import datetime
from fpdf import FPDF

from data_preparation import load_and_prepare_data
from train_model import train_model, predict_cost

# Constants
DATASET_PATH = "D:\\health_insurance\\content\\insurance.csv"
USER_FILE = "users.json"
CSV_FILENAME = "predictions_history.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password_strength(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None

def load_users():
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": {}}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def registration_page():
    st.title("📝 Register")
    with st.form("registration_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if username and password and confirm_password:
                if password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    validation = validate_password_strength(password)
                    if validation:
                        st.error(validation)
                    else:
                        users = load_users()
                        if username in users["users"]:
                            st.error("Username already exists!")
                        else:
                            users["users"][username] = hash_password(password)
                            save_users(users)
                            st.success("Registration successful!")
            else:
                st.error("All fields are required!")

def login_page():
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username or not password:
                st.error("Both fields are required!")
                return

            users = load_users()
            if username not in users["users"]:
                st.error("Username does not exist!")
                return

            if users["users"][username] == hash_password(password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Incorrect password!")

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Health Insurance Cost Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    for key, value in data.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    filename = f"prediction_report_{data['Username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def prediction_page():
    st.title("📊 Insurance Cost Prediction")

    st.sidebar.header(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.update({"authenticated": False, "username": None})
        st.experimental_rerun()

    st.write("## Enter Details:")

    with st.form("prediction_form"):
        name = st.text_input("Full Name")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 100, step=1)
            sex = st.selectbox("Sex", ["Male", "Female"])
            bmi = st.number_input("BMI", 0.0, 60.0, step=0.1)
            children = st.number_input("Number of Children", 0, 10, step=1)

        with col2:
            smoker = st.selectbox("Smoker", ["Yes", "No"])
            region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])
            exercise = st.number_input("Exercise Days/Week", 0, 7)
            sleep = st.number_input("Sleep Hours/Day", 0, 24)

        col3, col4 = st.columns(2)
        with col3:
            alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
            family_history = st.selectbox("Family Disease History", ["Yes", "No"])
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        with col4:
            diet = st.selectbox("Diet Type", ["Balanced", "Junk", "High Protein"])

        predict_button = st.form_submit_button("Predict Cost")

    if predict_button:
        sex_encoded = 0 if sex == "Male" else 1
        smoker_encoded = 0 if smoker == "Yes" else 1
        region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
        region_encoded = region_map[region]

        input_data = (age, sex_encoded, bmi, children, smoker_encoded, region_encoded)

        try:
            if 'regressor' not in st.session_state:
                X, Y = load_and_prepare_data(DATASET_PATH)
                regressor, r2_train, r2_test = train_model(X, Y)
                st.session_state['regressor'] = regressor

            cost = predict_cost(st.session_state['regressor'], input_data)
            st.success(f"Predicted Medical Insurance Cost: **Rs. {cost:.2f}**")

            prediction_info = {
                "Full Name": name,
                "Username": st.session_state["username"],
                "Age": age,
                "Sex": sex,
                "BMI": bmi,
                "Children": children,
                "Smoker": smoker,
                "Region": region,
                "Alcohol Consumption": alcohol,
                "Family Disease History": family_history,
                "Stress Level": stress,
                "Exercise Days": exercise,
                "Sleep Hours": sleep,
                "Diet Type": diet,
                "Predicted Cost": cost,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            df_new = pd.DataFrame([prediction_info])
            if os.path.exists(CSV_FILENAME):
                try:
                    df_existing = pd.read_csv(CSV_FILENAME)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except pd.errors.EmptyDataError:
                    # CSV file exists but empty
                    df_combined = df_new
                df_combined.to_csv(CSV_FILENAME, index=False)
            else:
                df_new.to_csv(CSV_FILENAME, index=False)

            st.success("Prediction data saved ✅")

            pdf_filename = generate_pdf(prediction_info)
            with open(pdf_filename, "rb") as f:
                st.download_button("📄 Download Report (PDF)", f, file_name=pdf_filename, mime="application/pdf")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

def admin_page():
    st.title("📋 Admin Panel")

    if os.path.exists(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
        except pd.errors.EmptyDataError:
            st.warning("Prediction history file is empty!")
            return

        if df.empty:
            st.warning("No records found in prediction history.")
            return

        # Add index for selection
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Record ID"}, inplace=True)

        desired_columns = [
            "Record ID",
            "Full Name", "Age", "Sex", "BMI", "Smoker", "Region", "Children",
            "Alcohol Consumption", "Family Disease History", "Stress Level",
            "Exercise Days", "Sleep Hours", "Diet Type", "Predicted Cost", "Timestamp"
        ]

        # Filter columns that exist in the dataframe
        available_columns = [col for col in desired_columns if col in df.columns]
        df_display = df[available_columns]

        st.dataframe(df_display, use_container_width=True)

        st.write("### Delete Records")

        # Multiselect for record IDs to delete
        to_delete = st.multiselect(
            "Select records to delete by Record ID:",
            options=df["Record ID"].tolist()
        )

        if st.button("Delete Selected Records"):
            if to_delete:
                df = df[~df["Record ID"].isin(to_delete)]
                # Drop the Record ID before saving
                df.drop(columns=["Record ID"], inplace=True)
                df.to_csv(CSV_FILENAME, index=False)
                st.success(f"Deleted {len(to_delete)} record(s).")
                st.experimental_rerun()
            else:
                st.warning("No records selected for deletion.")
    else:
        st.warning("No predictions found!")

def main():
    st.set_page_config(page_title="Health Insurance Predictor", layout="centered")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None

    if not st.session_state["authenticated"]:
        choice = st.sidebar.selectbox("Choose Option", ["Login", "Register"])
        if choice == "Register":
            registration_page()
        else:
            login_page()
    else:
        page = st.sidebar.selectbox("Navigation", ["Predict Insurance Cost", "Admin Panel"])
        if page == "Predict Insurance Cost":
            prediction_page()
        elif page == "Admin Panel":
            admin_page()

if __name__ == "__main__":
    main()
