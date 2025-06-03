import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests

# --- Streamlit Page Config ---
st.set_page_config(page_title="Training Dashboard", layout="wide")
st.markdown("## 🚀 Training Roll Out Tracker - Designing Intelligent Agentic AI Systems with LLMs")
st.markdown("---")

# --- CONFIGURATION ---
GITHUB_USER = "NJ65-lang"
GITHUB_REPO = "training-data-repository-1"
BRANCH = "main"
BATCHES = ["batch_1", "batch_2"]
MAX_SCORE = 15

# --- Helper Functions ---
def github_url(batch, file):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/{batch}/{file}"

def load_excel(url):
    response = requests.get(url)
    return pd.read_excel(BytesIO(response.content)) if response.status_code == 200 else None

def clean_attendance(df):
    df.columns = ['S.No', 'Name'] + df.columns[2:].tolist()
    df = df.iloc[:, 1:].reset_index(drop=True)  # Drop S.No and reset

    # Check if a date column has all NaNs (i.e., that day hasn't occurred)
    for col in df.columns[1:]:
        if df[col].isna().all():
            df[col] = "YTD"
        else:
            df[col] = df[col].apply(lambda x: 'P' if str(x).strip().lower() == 'p' else 'A')

    return df

def clean_pretest(df):
    df = df[['Full name', 'Total points']].dropna()
    df.columns = ['Name', 'Score']
    df['Score %'] = (df['Score'] / MAX_SCORE * 100).round(2)
    return df

# --- UI: Select Batch ---
selected_batch = st.selectbox("📁 Select Training Batch", BATCHES)

# --- Load Data ---
att_url = github_url(selected_batch, "attendance.xlsx")
pre_url = github_url(selected_batch, "pretest.xlsx")
att_df = load_excel(att_url)
pre_df = load_excel(pre_url)

if att_df is not None and pre_df is not None:
    att_clean = clean_attendance(att_df)
    pre_clean = clean_pretest(pre_df)

    # Show daily attendance table
    st.markdown("### 📋 Daily Attendance Record (P = Present, A = Absent)")
    st.dataframe(att_clean, use_container_width=True)

    # Merge for score dashboard only
    summary_df = pd.merge(pre_clean, att_clean, on="Name", how="outer").dropna(subset=["Name"])

    # --- KPI Cards ---
    avg_score = summary_df['Score'].mean().round(2)
    total_participants = summary_df.shape[0]

    col1, col2 = st.columns(2)
    col1.metric("👥 Participants", total_participants)
    col2.metric("📊 Avg Pre-Test Score", f"{avg_score} / {MAX_SCORE}")

    # --- Pre-Test Score Chart ---
    st.markdown("### 📈 Pre-Test Score Distribution")
    chart_df = summary_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(chart_df["Name"], chart_df["Score"], color="mediumseagreen")
    ax.set_ylim(0, MAX_SCORE)
    ax.set_ylabel("Score")
    ax.set_title("Participant Pre-Test Scores (out of 15)")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

else:
    st.error("❌ Failed to load one or both Excel files. Please check the file names or GitHub URLs.")
