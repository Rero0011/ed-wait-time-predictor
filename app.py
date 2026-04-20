import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="ED Wait Time Predictor", layout="wide")

# -----------------------------
# Load model and metadata
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    explainer = shap.Explainer(model)
    return model, model_columns, explainer

model, model_columns, explainer = load_model()

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.header-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 28px 26px 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    backdrop-filter: blur(8px);
}

.main-title {
    color: white;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 10px 0;
    padding-top: 4px;
}

.subtle-divider {
    height: 1px;
    width: 100%;
    background: linear-gradient(90deg, rgba(34,197,94,0), rgba(34,197,94,0.35), rgba(34,197,94,0));
    margin: 8px 0 20px 0;
    border-radius: 999px;
}

.section-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 18px 18px 12px 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.prediction-card {
    background: linear-gradient(90deg, #052e16, #14532d);
    padding: 18px 22px;
    border-radius: 14px;
    border: 1px solid #166534;
    margin-top: 10px;
    margin-bottom: 18px;
    animation: fadeInUp 0.5s ease-out;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
}

.note-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: -6px;
    margin-bottom: 18px;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stButton > button {
    background: linear-gradient(90deg, #0f172a, #1e293b);
    color: white;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

.stButton > button:hover {
    border-color: #22c55e;
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.22);
}

label, .stMarkdown, .stText {
    color: #e5e7eb !important;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App header
# -----------------------------
st.markdown("""
<div class="header-card">
    <div style="color:#86efac; font-size:14px; font-weight:600; margin-bottom:8px;">
        Explainable AI Dashboard
    </div>
    <div class="main-title">
        Emergency Department Wait Time Predictor
    </div>
    <div style="color:#cbd5e1; font-size:16px; line-height:1.7;">
        Predict emergency department wait times using a trained <b>XGBoost</b> model and explore
        the key factors influencing each prediction through SHAP-based explanations.
    </div>
</div>
<div class="subtle-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------
# Helper functions
# -----------------------------
def build_input_row(
    urgency, region, season, time_of_day, day_of_week,
    nurse_ratio, specialist, facility, visit_month, visit_hour, visit_day
):
    input_dict = {col: 0 for col in model_columns}

    numeric_values = {
        "Nurse-to-Patient Ratio": nurse_ratio,
        "Specialist Availability": specialist,
        "Facility Size (Beds)": facility,
        "Visit Month": visit_month,
        "Visit Hour": visit_hour,
        "Visit Day": visit_day,
    }

    for col, value in numeric_values.items():
        if col in input_dict:
            input_dict[col] = value

    candidate_cols = [
        f"Region_{region}",
        f"Day of Week_{day_of_week}",
        f"Season_{season}",
        f"Time of Day_{time_of_day}",
        f"Urgency Level_{urgency}",
    ]

    for col in candidate_cols:
        if col in input_dict:
            input_dict[col] = 1

    return pd.DataFrame([input_dict])[model_columns]


def prettify_feature_name(name: str) -> str:
    return name.replace("_", " ")


def build_human_explanations(top_features, urgency, region, season, time_of_day, day_of_week):
    explanations = []

    for feat, val in top_features.items():
        clean_name = prettify_feature_name(feat)

        if "Urgency Level" in clean_name:
            if "Low" in clean_name and val > 0:
                explanations.append(
                    "• This patient was classified as **low urgency**, meaning more critical patients were prioritised first, which likely made the wait longer."
                )
            elif "Medium" in clean_name:
                explanations.append(
                    "• This patient was classified as **medium urgency**, which usually leads to a moderate wait depending on how busy the department is."
                )
            elif "High" in clean_name or "Critical" in clean_name:
                explanations.append(
                    "• This was a **high-priority case**, so the patient was likely seen faster, helping to reduce the wait."
                )

        elif "Day of Week" in clean_name:
            explanations.append(
                f"• The visit happened on **{day_of_week}**, which the model links to the typical demand level for that day."
            )

        elif "Season" in clean_name:
            explanations.append(
                f"• The visit took place during **{season}**, which can affect how busy the department is and therefore influence waiting time."
            )

        elif "Time of Day" in clean_name:
            if val > 0:
                explanations.append(
                    f"• Visiting during the **{time_of_day.lower()}** is often linked to busier periods, which may have made the wait longer."
                )
            else:
                explanations.append(
                    f"• Visiting during the **{time_of_day.lower()}** is often linked to quieter periods, which may have helped keep the wait shorter."
                )

        elif "Nurse-to-Patient Ratio" in clean_name:
            if val > 0:
                explanations.append(
                    "• The staffing level suggested more pressure on available nurses, which may have contributed to a longer wait."
                )
            else:
                explanations.append(
                    "• The staffing level suggested less pressure on available nurses, which may have helped reduce the wait."
                )

        elif "Facility Size" in clean_name:
            explanations.append(
                "• The size of the hospital was one of the factors the model used when estimating the expected waiting time."
            )

        elif "Specialist Availability" in clean_name:
            explanations.append(
                "• Specialist availability had some influence on the estimate, although it was not one of the strongest factors."
            )

        elif "Region" in clean_name:
            explanations.append(
                f"• The **{region.lower()}** location also contributed to the estimate."
            )

        elif "Visit Hour" in clean_name:
            explanations.append(
                "• The exact hour of arrival also influenced the estimate."
            )

        elif "Visit Month" in clean_name:
            explanations.append(
                "• The month of the visit contributed to the estimate, reflecting seasonal patterns in demand."
            )

        elif "Visit Day" in clean_name:
            explanations.append(
                "• The day of the month had a smaller influence on the final estimate."
            )

    # Remove duplicates while preserving order
    deduped = []
    seen = set()
    for line in explanations:
        if line not in seen:
            deduped.append(line)
            seen.add(line)

    return deduped[:3]


# -----------------------------
# Inputs section
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    urgency = st.selectbox("Urgency Level", ["Low", "Medium", "High", "Critical"])
    region = st.selectbox("Region", ["Urban", "Rural"])
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

with col2:
    time_of_day = st.selectbox(
        "Time of Day",
        ["Early Morning", "Late Morning", "Afternoon", "Evening", "Night"]
    )
    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    nurse_ratio = st.slider("Nurse-to-Patient Ratio", 1, 5, 3)

with col3:
    specialist = st.slider("Specialist Availability", 0, 10, 3)
    facility = st.slider("Facility Size (Beds)", 10, 200, 80)
    visit_hour = st.slider("Visit Hour", 0, 23, 12)

col4, col5 = st.columns(2)
with col4:
    visit_month = st.slider("Visit Month", 1, 12, 6)
with col5:
    visit_day = st.slider("Visit Day", 1, 31, 15)

predict_clicked = st.button("Predict Wait Time", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if predict_clicked:
    input_df = build_input_row(
        urgency, region, season, time_of_day, day_of_week,
        nurse_ratio, specialist, facility, visit_month, visit_hour, visit_day
    )

    prediction = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div class="prediction-card">
            <div style="color:#bbf7d0; font-size:16px; margin-bottom:6px;">Predicted Wait Time</div>
            <div style="color:white; font-size:34px; font-weight:800;">{prediction:.1f} minutes</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="note-card">
            <div style="color:#cbd5e1; font-size:15px; line-height:1.6;">
                This estimate is based on patterns learned from historical data and the information entered above.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    shap_values = explainer(input_df)
    shap_series = pd.Series(shap_values.values[0], index=model_columns)

    top_features = shap_series.reindex(
        shap_series.abs().sort_values(ascending=False).index
    ).head(5)

    # Chart section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top 5 Prediction Drivers")

    display_names = [prettify_feature_name(f) for f in top_features.index[::-1]]
    display_values = top_features.values[::-1]
    colors = ["#22c55e" if v > 0 else "#3b82f6" for v in display_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=display_values,
        y=display_names,
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.2f}<extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1, line_color="#94a3b8", opacity=0.7)

    fig.update_layout(
        title="Top 5 Prediction Drivers",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="white", size=13),
        xaxis=dict(
            title="SHAP Value (impact on predicted wait time)",
            gridcolor="#1e293b",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            showgrid=False
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Interpretation section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Why this wait time was predicted")

    st.markdown(
        "Here are the main factors that influenced this estimate:"
    )

    human_explanations = build_human_explanations(
        top_features, urgency, region, season, time_of_day, day_of_week
    )

    for line in human_explanations:
        st.markdown(line)

    st.markdown(
        "<span style='color:#94a3b8;'>Green bars show factors that likely made the wait longer, while blue bars show factors that helped keep it shorter.</span>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)