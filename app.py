"""
app.py
======
Streamlit application for Customer Churn Prediction.

This UI collects raw customer information, sends it through the exact
same validation + preprocessing + model pipeline used in training
(src.prediction.predict_customer -> the saved joblib Pipeline), and
displays the model's own prediction and probability. There is no
second, rule-based prediction logic anywhere in this file.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prediction import ValidationError, load_model_and_metadata, predict_customer

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load model (cached so we do NOT retrain on every click)
# ----------------------------------------------------------------------


@st.cache_resource
def get_model_and_metadata():
    return load_model_and_metadata()


try:
    model, metadata = get_model_and_metadata()
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)


def allowed(field: str, default: list[str]) -> list[str]:
    values = metadata.get("categorical_allowed_values", {}).get(field)
    return values if values else default


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("📉 Customer Churn Prediction")
st.caption(
    "Predicts the likelihood that a customer will churn, using a trained "
    "machine-learning model — not hard-coded business rules."
)

if not MODEL_LOADED:
    st.error(
        f"Could not load the trained model: {LOAD_ERROR}\n\n"
        "Run `python train.py` first to train and save the model, then relaunch this app."
    )
    st.stop()

tab_predict, tab_about = st.tabs(["🔮 Predict", "ℹ️ About the Model"])

# ----------------------------------------------------------------------
# PREDICT TAB
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Customer Information")

    with st.form("churn_form"):
        st.markdown("**Personal Information**")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", allowed("gender", ["Female", "Male"]))
        senior = c2.selectbox("Senior Citizen", ["No", "Yes"])
        partner = c3.selectbox("Partner", allowed("Partner", ["No", "Yes"]))
        dependents = c4.selectbox("Dependents", allowed("Dependents", ["No", "Yes"]))

        st.markdown("**Account Information**")
        c1, c2, c3, c4 = st.columns(4)
        tenure = c1.number_input("Tenure (months)", min_value=0, max_value=130, value=12, step=1)
        contract = c2.selectbox(
            "Contract", allowed("Contract", ["Month-to-month", "One year", "Two year"])
        )
        payment_method = c3.selectbox(
            "Payment Method",
            allowed(
                "PaymentMethod",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            ),
        )
        paperless = c4.selectbox("Paperless Billing", allowed("PaperlessBilling", ["No", "Yes"]))

        st.markdown("**Services**")
        c1, c2, c3 = st.columns(3)
        phone_service = c1.selectbox("Phone Service", allowed("PhoneService", ["No", "Yes"]))
        multiple_lines = c2.selectbox(
            "Multiple Lines", allowed("MultipleLines", ["No phone service", "No", "Yes"])
        )
        internet_service = c3.selectbox(
            "Internet Service", allowed("InternetService", ["DSL", "Fiber optic", "No"])
        )

        c1, c2, c3 = st.columns(3)
        online_security = c1.selectbox(
            "Online Security", allowed("OnlineSecurity", ["No", "Yes", "No internet service"])
        )
        online_backup = c2.selectbox(
            "Online Backup", allowed("OnlineBackup", ["No", "Yes", "No internet service"])
        )
        device_protection = c3.selectbox(
            "Device Protection", allowed("DeviceProtection", ["No", "Yes", "No internet service"])
        )

        c1, c2, c3 = st.columns(3)
        tech_support = c1.selectbox(
            "Tech Support", allowed("TechSupport", ["No", "Yes", "No internet service"])
        )
        streaming_tv = c2.selectbox(
            "Streaming TV", allowed("StreamingTV", ["No", "Yes", "No internet service"])
        )
        streaming_movies = c3.selectbox(
            "Streaming Movies", allowed("StreamingMovies", ["No", "Yes", "No internet service"])
        )

        st.markdown("**Financial Information**")
        c1, c2 = st.columns(2)
        monthly_charges = c1.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=70.0, step=0.5
        )
        total_charges = c2.number_input(
            "Total Charges ($)", min_value=0.0, max_value=1_000_000.0, value=840.0, step=1.0
        )

        submitted = st.form_submit_button("🔮 PREDICT CHURN", use_container_width=True)

    if submitted:
        raw_input = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            with st.spinner("Running the trained model..."):
                result = predict_customer(raw_input, model, metadata)
        except ValidationError as e:
            st.error(f"Please fix the following before predicting:\n\n{e}")
        except Exception as e:  # noqa: BLE001 - surface unexpected errors safely
            st.error(f"Something went wrong while generating the prediction: {e}")
        else:
            st.divider()
            st.subheader("Prediction Result")

            is_churn = result["prediction"] == 1
            churn_prob = result.get("churn_probability")

            col1, col2, col3 = st.columns(3)
            with col1:
                if is_churn:
                    st.error(f"**Prediction:**\n\n🔴 {result['prediction_label'].upper()}")
                else:
                    st.success(f"**Prediction:**\n\n🟢 {result['prediction_label'].upper()}")

            if result["probabilities"]:
                no_prob = result["probabilities"].get("0", 1 - (churn_prob or 0))
                with col2:
                    st.metric("Churn Probability", f"{(churn_prob or 0) * 100:.1f}%")
                with col3:
                    st.metric("No-Churn Probability", f"{no_prob * 100:.1f}%")

                st.progress(min(max(churn_prob or 0, 0.0), 1.0))

            st.markdown("**Interpretation**")
            if is_churn:
                st.write(
                    f"According to the trained machine-learning model, this customer has a "
                    f"relatively high predicted probability of churn "
                    f"({(churn_prob or 0) * 100:.1f}%) based on the information provided. "
                    "The model predicts that this customer is **likely to churn** — this is a "
                    "prediction, not a certainty."
                )
            else:
                st.write(
                    f"According to the trained machine-learning model, this customer has a "
                    f"relatively low predicted probability of churn "
                    f"({(churn_prob or 0) * 100:.1f}%) based on the information provided. "
                    "The model predicts that this customer is **not likely to churn** — this is "
                    "a prediction, not a certainty."
                )

            with st.expander("Model-associated factors (feature importance)"):
                st.caption(
                    "These are the features the trained model relies on most heavily overall "
                    "(not specific to this one prediction, and not a causal claim)."
                )
                try:
                    inner_model = model.named_steps["model"]
                    preprocessor = model.named_steps["preprocessor"]
                    feature_names = preprocessor.get_feature_names_out()
                    if hasattr(inner_model, "feature_importances_"):
                        importances = inner_model.feature_importances_
                        imp_df = (
                            pd.DataFrame({"feature": feature_names, "importance": importances})
                            .sort_values("importance", ascending=False)
                            .head(10)
                        )
                        st.bar_chart(imp_df.set_index("feature"))
                    else:
                        st.write("Feature importance is not available for this model type.")
                except Exception:
                    st.write("Feature importance is not available for this model.")

            st.button("Make Another Prediction", on_click=lambda: None)

# ----------------------------------------------------------------------
# ABOUT TAB
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("About the Model")

    m = metadata
    test_metrics = m.get("test_metrics", {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model type**")
        st.write(m.get("model_name", "n/a"))
        st.markdown("**Training dataset**")
        st.write(m.get("dataset_source", "n/a"))
        st.markdown("**Training samples**")
        st.write(m.get("n_train_samples", "n/a"))
        st.markdown("**Test samples**")
        st.write(m.get("n_test_samples", "n/a"))
        st.markdown("**Trained on**")
        st.write(m.get("training_date", "n/a"))

    with col2:
        st.markdown("**Test-set performance**")
        st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        st.metric("F1-score", f"{test_metrics.get('f1', 0):.3f}")
        met_c1, met_c2, met_c3 = st.columns(3)
        met_c1.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
        met_c2.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        met_c3.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")

    st.info(
        "This model was selected using 5-fold cross-validated ROC-AUC across several "
        "candidate algorithms (not accuracy alone), then hyperparameter-tuned. "
        "Metrics above are computed once on a held-out test set the model never saw "
        "during training or tuning."
    )
