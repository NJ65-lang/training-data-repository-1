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
    # Set proper column headers
    df.columns = ['S.No', 'Name'] + df.columns[2:].tolist()
    df = df.iloc[:, 1:]  # Drop S.No

    # Remove empty rows
    df = df[df['Name'].notna() & (df['Name'].astype(str).str.strip() != '')].reset_index(drop=True)

    # Convert attendance status
    for col in df.columns[1:]:
        if df[col].isna().all():
            df[col] = "YTD"
        else:
            df[col] = df[col].apply(lambda x: 'P' if str(x).strip().lower() == 'p' else 'A')

    # Add new S.No column
    df.insert(0, 'S.No', range(1, len(df) + 1))
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

    # --- Daily Attendance Table ---
    st.markdown("### 📋 Daily Attendance Record")
    st.dataframe(att_clean, use_container_width=True)
    st.caption("🔹 **Legend:** P = Present, A = Absent, YTD = Yet to be Delivered")

    # --- Summary for Pre-test + Attendance (optional for score chart) ---
    summary_df = pd.merge(pre_clean, att_clean[['Name']], on="Name", how="right")

    avg_score = summary_df['Score'].mean().round(2)
    total_participants = summary_df.shape[0]

    col1, col2 = st.columns(2)
    col1.metric("👥 Participants", total_participants)
    col2.metric("📊 Avg Pre-Test Score", f"{avg_score} / {MAX_SCORE}")

    # --- Pre-Test Score Chart ---
  # --- Pre-Test Score Chart (Improved with color coding) ---
st.markdown("### 📈 Pre-Test Scores (Top 3 in Green, Bottom 3 in Red)")

# Sort and color
chart_df = summary_df.dropna(subset=["Score"]).sort_values(by="Score", ascending=False).reset_index(drop=True)
colors = [
    "green" if i < 3 else "red" if i >= len(chart_df) - 3 else "skyblue"
    for i in range(len(chart_df))
]

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(chart_df["Name"], chart_df["Score"], color=colors)

# Labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval}', ha='center', va='bottom')

# Styling
ax.set_ylim(0, MAX_SCORE)
ax.set_yticks(range(0, MAX_SCORE + 1))
ax.set_ylabel("Score")
ax.set_title("🏆 Pre-Test Scores")
plt.xticks(rotation=45)

st.pyplot(fig)

else:
    st.error("❌ Failed to load one or both Excel files. Please check the file names or GitHub URLs.")
