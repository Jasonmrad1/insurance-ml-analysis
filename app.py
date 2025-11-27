import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Healthcare Predictor",
    layout="wide"
)

df = pd.read_csv("data/medical_insurance.csv")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        color: white;
    }
    .main-header {
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load("models/annual_medical_cost_model.pkl")
        reg_features = joblib.load("features/annual_medical_cost_features.pkl")
        clf_model = joblib.load("models/risk_score_model.pkl")
        clf_features = joblib.load("features/risk_score_features.pkl")
        lr_premium = joblib.load("models/lr_premium.pkl")
        lr_features = joblib.load("features/lr_premium_features.pkl")
        return reg_model, reg_features, clf_model, clf_features, lr_premium, lr_features
    except Exception as e:
        st.error(f"MODEL LOAD ERROR: {e}")
        return None, None, None, None, None, None

reg_model, reg_features, clf_model, clf_features, lr_premium, lr_features = load_models()

yes_no_map = lambda x: 1 if x == "Yes" else 0
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Annual Medical Cost", "Risk Score"])

if page == "Dashboard":
    st.markdown('<h1 style="color:#ffffff;">Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#ffffff;">Key Metrics</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_cost = round(df['annual_medical_cost'].mean(), 2)
        st.metric("Average Medical Cost", f"${avg_cost:,.2f}", "+5.2%")

    with col2:
        high_risk_pct = 36.7  # or calculate from your data
        st.metric("High Risk Patients", f"{high_risk_pct:.2f}%", "-2.1%")

    with col3:
        total_patients = len(df)  # or your exact number
        st.metric("Total Patients", f"{total_patients:,}", "+8%")

    with col4:
        avg_risk = round(df['risk_score'].mean(), 2)
        st.metric("Average Risk Score", f"{avg_risk:.2f}", "-1.2%")

    st.markdown("---")
    st.markdown('<h3 style="color:#ffffff;">Cost & Risk Analysis</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        costs = df['annual_medical_cost'].dropna()
        costs = costs[costs <= 30000]
        fig = px.histogram(
            x=costs,
            nbins=40,
            title="Medical Cost Distribution (up to $30k)",
            color_discrete_sequence=['#667eea'],
            template='plotly_white'
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit, sans-serif", color="#f1f1f1"),
            title_font=dict(family="Outfit, sans-serif", size=16, color="#dedede"),
            xaxis=dict(title="Annual Medical Cost ($)", range=[0, 30000])
        )
        
        fig.update_traces(marker_line_width=0.5, marker_line_color="white")
        st.plotly_chart(fig, width='stretch')

    with col2:
        values = [10516, 24523, 28178, 20829, 15944]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
        fig = px.pie(
            values=values,
            names=labels,
            title="Patient Risk Distribution",
            color_discrete_sequence=["#90A9FF",  "#A6BBFF", "#B0C7FF", "#4F7CD6", "#344E99"],
            template='plotly_white'
        )

        fig.update_traces(
            pull=[0.005]*len(values),
            textinfo='percent+label',
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit, sans-serif", color="#f1f1f1"),
            title_font=dict(family="Outfit, sans-serif", size=16, color="#dedede")
        )

        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown('<h3 style="color:#ffffff;">Risk & Cost Drivers</h3>', unsafe_allow_html=True)
    risk_factors = [
        'age', 'chronic_count', 'systolic_bp', 'bmi', 'smoker', 
        'diastolic_bp', 'hypertension', 'annual_premium', 'hba1c',
        'monthly_premium', 'total_claims_paid', 'mental_health', 
        'arthritis', 'avg_claim_amount', 'visits_last_year'
    ]
    importance = [
        27363, 11419, 9018, 8999, 6627, 
        3493, 2770, 2447, 2178, 1824, 
        1646, 1617, 1300, 1104, 1056
    ]

    fig_risk = px.bar(
        x=importance,
        y=risk_factors,
        orientation='h',
        title="Top Risk Factors (High Risk Patients)",
        color=importance,
        color_continuous_scale=['#5C9DED', '#4A82D1', '#3665B5', '#244A99', '#122D7D'],
        template='plotly_white'
    )

    fig_risk.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", color="#f1f1f1"),
        title_font=dict(family="Outfit, sans-serif", size=16, color="#dedede"),
        xaxis_title="Feature Importance",
        yaxis_title="Factors",
        margin=dict(l=150, r=40, t=50, b=40)
    )

    st.plotly_chart(fig_risk, width='stretch')
    cost_factors = [
        'monthly_premium', 'total_claims_paid', 'avg_claim_amount', 'bmi', 'risk_score',
        'ldl', 'days_hospitalized_last_3yrs', 'age', 'systolic_bp', 'diastolic_bp',
        'chronic_count', 'claims_count', 'visits_last_year', 'smoker', 'medication_count'
    ]
    importance_cost = [
        14491, 8325, 6396, 5403, 5252,
        4939, 4926, 4140, 3952, 3510,
        2578, 2489, 2444, 1948, 1399
    ]

    fig_cost = px.bar(
        x=importance_cost,
        y=cost_factors,
        orientation='h',
        title="Top Features for Annual Medical Cost",
        color=importance_cost,
        color_continuous_scale=['#A6C8FF', '#7FB3FF', '#5599FF', '#2D7FFF', '#125FCC'], 
        template='plotly_white'
    )

    fig_cost.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", color="#f1f1f1"),
        title_font=dict(family="Outfit, sans-serif", size=16, color="#dedede"),
        xaxis_title="Feature Importance",
        yaxis_title="Factors",
        margin=dict(l=200, r=40, t=50, b=40)  # wider left margin for long labels
    )

    st.plotly_chart(fig_cost, width='stretch')

elif page == "Annual Medical Cost":
    st.header("Annual Medical Cost Prediction")

    with st.expander("Patient Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 120, 0, step=1)
            bmi = st.number_input("BMI", 0.0, 60.0, 0.0, step=0.01, format="%.2f")
        with col2:
            smoker = st.selectbox("Smoker", ["Never", "Former", "Current"])
            chronic_count = st.number_input("Chronic Conditions Count", 0, 50, 0, step=1)
        with col3:
            risk_score = st.number_input("Risk Score", min_value=0.0, max_value=1.0, value=0.0, format="%.4f")

    with st.expander("Vitals & Labs", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic BP", 0, 250, 0, step=1)
            diastolic_bp = st.number_input("Diastolic BP", 0, 150, 0, step=1)
        with col2:
            ldl = st.number_input("LDL Cholesterol", 0.0, 300.0, 0.0, step=0.1, format="%.1f")

    with st.expander("Visits & Procedures", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            visits_last_year = st.number_input("Visits Last Year", 0, 50, 0, step=1)
            days_hospitalized_last_3yrs = st.number_input("Days Hospitalized Last 3 Years", 0, 500, 0, step=1)
        with col2:
            medication_count = st.number_input("Medication Count", 0, 50, 0, step=1)
            proc_imaging_count = st.number_input("Imaging Procedures Count", 0, 50, 0, step=1)
        with col3:
            proc_physio_count = st.number_input("Physio Procedures Count", 0, 50, 0, step=1)
            proc_consult_count = st.number_input("Consult Procedures Count", 0, 50, 0, step=1)
            proc_lab_count = st.number_input("Lab Procedures Count", 0, 50, 0, step=1)
            had_major_procedure = st.selectbox("Had Major Procedure", ["No", "Yes"])
            had_major_procedure_encoded = 1 if had_major_procedure=="Yes" else 0

    with st.expander("Medical Conditions", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            asthma = st.selectbox("Asthma", ["No", "Yes"])
        with col2:
            copd = st.selectbox("COPD", ["No", "Yes"])
            cardiovascular_disease = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
            cancer_history = st.selectbox("Cancer History", ["No", "Yes"])
        with col3:
            kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
            liver_disease = st.selectbox("Liver Disease", ["No", "Yes"])
            arthritis = st.selectbox("Arthritis", ["No", "Yes"])
            mental_health = st.selectbox("Mental Health Issues", ["No", "Yes"])

        def yes_no(val):
            return 1 if val=="Yes" else 0

    with st.expander("Financials & Claims", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_premium = st.number_input("Monthly Premium", 0.0, 10000.0, 0.0, step=0.01, format="%.2f")
        with col2:
            claims_count = st.number_input("Claims Count", 0, 100, 0, step=1)
            avg_claim_amount = st.number_input("Average Claim Amount", 0.0, 100000.0, 0.0, step=0.01, format="%.2f")
        with col3:
            total_claims_paid = st.number_input("Total Claims Paid", 0.0, 1000000.0, 0.0, step=0.01, format="%.2f")

    inputs = {
        "age": age,
        "bmi": bmi,
        "smoker": {"Never":0, "Former":1, "Current":2}[smoker],
        "chronic_count": chronic_count,
        "risk_score": risk_score,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "ldl": ldl,
        "visits_last_year": visits_last_year,
        "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
        "medication_count": medication_count,
        "proc_imaging_count": proc_imaging_count,
        "proc_physio_count": proc_physio_count,
        "proc_consult_count": proc_consult_count,
        "proc_lab_count": proc_lab_count,
        "had_major_procedure": had_major_procedure_encoded,
        "hypertension": yes_no(hypertension),
        "diabetes": yes_no(diabetes),
        "asthma": yes_no(asthma),
        "copd": yes_no(copd),
        "cardiovascular_disease": yes_no(cardiovascular_disease),
        "cancer_history": yes_no(cancer_history),
        "kidney_disease": yes_no(kidney_disease),
        "liver_disease": yes_no(liver_disease),
        "arthritis": yes_no(arthritis),
        "mental_health": yes_no(mental_health),
        "claims_count": claims_count,
        "avg_claim_amount": avg_claim_amount,
        "total_claims_paid": total_claims_paid
    }

    X_user = pd.DataFrame([{feat: inputs.get(feat, 0) for feat in lr_features}])
    monthly_premium_residual = monthly_premium - lr_premium.predict(X_user)[0]
    monthly_premium_residual *= 0.5
    inputs["monthly_premium_residual"] = monthly_premium_residual

    df_reg = pd.DataFrame([inputs])[reg_features]

    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #667eea !important;
            color: white !important;
            width: 100% !important;
            height: 40px !important;
            border-radius: 5px !important;
            border: none !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #556cd6 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        predict_clicked = st.button("Predict", key="predict_cost", use_container_width=True)

    if predict_clicked:
        pred = reg_model.predict(df_reg)[0]

        st.markdown(f"""
        <div style="
            background-color:#333333;
            padding:20px;
            border-radius:10px;
            color:white;
            margin-bottom:20px;
        ">
            <h3 style="margin-bottom:10px; color:#f1f1f1;">Prediction Result</h3>
            <p style="font-size:20px; color:#00ff99; margin-bottom:5px;">
                Predicted Annual Medical Cost: <strong>${pred:.2f}</strong>
            </p>
            <p style="color:#aaaaaa; font-size:12px;">Values are based on patient features</p>
        </div>
        """, unsafe_allow_html=True)

        feature_order = df_reg.iloc[0].sort_values(ascending=False)
        fig = px.bar(
            x=feature_order.index,
            y=feature_order.values,
            labels={"x": "Feature", "y": "Value"},
            title="Patient Feature Values",
            color=feature_order.values,
            color_continuous_scale=["#667eea", "#334f8d"],
        )
        fig.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font_color="white",
            title=dict(x=0.5, xanchor='center'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("<h3 style='color:white;'>Input Summary</h3>", unsafe_allow_html=True)
        float_cols = df_reg.select_dtypes(include='float').columns
        format_dict = {col: "{:.2f}" for col in float_cols}
        st.dataframe(
            df_reg.style
            .format(format_dict)
            .background_gradient(cmap='Greys', axis=1)
            .set_properties(**{'background-color': '#333333', 'color': 'white'})
        )

elif page == "Risk Score":
    st.header("Risk Score Prediction")

    with st.expander("Patient Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 120, 0, step=1)
            bmi = st.number_input("BMI", 0.0, 60.0, 0.0, step=0.01, format="%.2f")
        with col2:
            smoker = st.selectbox("Smoker", ["Never", "Former", "Current"], key="smoker_r")
            chronic_count = st.number_input("Chronic Conditions Count", 0, 50, 0, step=1)
        with col3:
            risk_score = st.number_input("Risk Score", 0.0, 1.0, 0.0, step=0.01, format="%.4f")

    with st.expander("Vitals & Labs", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic BP", 0.0, 250.0, 0.0, step=0.1, format="%.1f")
            diastolic_bp = st.number_input("Diastolic BP", 0.0, 150.0, 0.0, step=0.1, format="%.1f")
        with col2:
            hba1c = st.number_input("HbA1c (%)", 0.0, 15.0, 0.0, step=0.1, format="%.1f")

    with st.expander("Visits & Procedures", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            visits_last_year = st.number_input("Visits Last Year", 0, 50, 0, step=1)
            medication_count = st.number_input("Medication Count", 0, 50, 0, step=1)
        with col2:
            proc_imaging_count = st.number_input("Imaging Procedures Count", 0, 50, 0, step=1)
            proc_surgery_count = st.number_input("Surgery Procedures Count", 0, 50, 0, step=1)
        with col3:
            proc_lab_count = st.number_input("Lab Procedures Count", 0, 50, 0, step=1)
            had_major_procedure = st.selectbox("Had Major Procedure", ["No", "Yes"], key="major")

    with st.expander("Medical Conditions", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            asthma = st.selectbox("Asthma", ["No", "Yes"])
        with col2:
            copd = st.selectbox("COPD", ["No", "Yes"])
            cardiovascular_disease = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
            cancer_history = st.selectbox("Cancer History", ["No", "Yes"])
        with col3:
            kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
            liver_disease = st.selectbox("Liver Disease", ["No", "Yes"])
            arthritis = st.selectbox("Arthritis", ["No", "Yes"])
            mental_health = st.selectbox("Mental Health Issues", ["No", "Yes"])

    with st.expander("Financials & Claims", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            annual_premium = st.number_input("Annual Premium", 0.0, 100000.0, 0.0, step=0.01, format="%.2f")
        with col2:
            claims_count = st.number_input("Claims Count", 0, 100, 0, step=1)
            avg_claim_amount = st.number_input("Average Claim Amount", 0.0, 100000.0, 0.0, step=0.01, format="%.2f")
        with col3:
            total_claims_paid = st.number_input("Total Claims Paid", 0.0, 1000000.0, 0.0, step=0.01, format="%.2f")

    inputs = {
        "age": age,
        "bmi": bmi,
        "smoker": {"Never": 0, "Former": 1, "Current": 2}[smoker],
        "chronic_count": chronic_count,
        "risk_score": risk_score,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "hba1c": hba1c,
        "visits_last_year": visits_last_year,
        "medication_count": medication_count,
        "proc_imaging_count": proc_imaging_count,
        "proc_surgery_count": proc_surgery_count,
        "proc_lab_count": proc_lab_count,
        "had_major_procedure": 1 if had_major_procedure=="Yes" else 0,
        "hypertension": 1 if hypertension=="Yes" else 0,
        "diabetes": 1 if diabetes=="Yes" else 0,
        "asthma": 1 if asthma=="Yes" else 0,
        "copd": 1 if copd=="Yes" else 0,
        "cardiovascular_disease": 1 if cardiovascular_disease=="Yes" else 0,
        "cancer_history": 1 if cancer_history=="Yes" else 0,
        "kidney_disease": 1 if kidney_disease=="Yes" else 0,
        "liver_disease": 1 if liver_disease=="Yes" else 0,
        "arthritis": 1 if arthritis=="Yes" else 0,
        "mental_health": 1 if mental_health=="Yes" else 0,
        "annual_premium": annual_premium,
        "claims_count": claims_count,
        "avg_claim_amount": avg_claim_amount,
        "total_claims_paid": total_claims_paid
    }

    df_clf = pd.DataFrame([inputs])[clf_features]

    col_left, col_right = st.columns([3, 1])
    with col_right:
        st.markdown(
            """
            <style>
            .stButton > button {
                background-color: #667eea !important;
                color: white !important;
                width: 100% !important;
                height: 40px !important;
                border-radius: 5px !important;
                border: none !important;
                font-weight: 500 !important;
            }
            .stButton > button:hover {
                background-color: #556cd6 !important;
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        predict_clicked = st.button("Predict", key="predict_risk", use_container_width=True)

    if predict_clicked:
        pred_class = clf_model.predict(df_clf)[0]

        color_map = {
            "Very Low": "#00ff99",   # green
            "Low": "#00ff99",        # green
            "Medium": "#ffaa00",     # orange
            "High": "#ff5555",       # red
            "Very High": "#ff5555"   # red
        }
        pred_color = color_map.get(pred_class, "#00ff99")

        st.markdown(
            f"""
            <div style="
                background-color:#333333;
                padding:20px;
                border-radius:10px;
                color:white;
                margin-bottom:20px;
            ">
                <h3 style="margin-bottom:10px; color:#f1f1f1;">Prediction Result</h3>
                <p style="font-size:20px; color:{pred_color}; margin-bottom:5px;">
                    Predicted Risk Class: <strong>{pred_class}</strong>
                </p>
                <p style="color:#aaaaaa; font-size:12px;">Values are based on patient features</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        feature_order = df_clf.iloc[0].sort_values(ascending=False)
        fig = px.bar(
            x=feature_order.index,
            y=feature_order.values,
            labels={"x": "Feature", "y": "Value"},
            title="Patient Feature Values",
            color=feature_order.values,
            color_continuous_scale=["#667eea", "#334f8d"]
        )
        fig.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font_color="white",
            title=dict(x=0.5, xanchor='center'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("<h3 style='color:white;'>Input Summary</h3>", unsafe_allow_html=True)
        float_cols = df_clf.select_dtypes(include='float').columns
        format_dict = {col: "{:.2f}" for col in float_cols}

        st.dataframe(
            df_clf.style
            .format(format_dict)  # apply rounding only to float columns
            .background_gradient(cmap='Greys', axis=1)
            .set_properties(**{
                'background-color': '#333333',
                'color': 'white'
            })
        )

st.markdown("---")
st.caption("Healthcare Prediction Dashboard • Using machine learning for medical cost and risk prediction")