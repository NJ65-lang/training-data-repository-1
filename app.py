import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests

st.set_page_config(page_title="Training Dashboard", layout="wide")
st.title("📊 Microsoft AXP Team Training Rollout")

# --- CONFIGURATION ---
GITHUB_USER = "NJ65-lang"
GITHUB_REPO = "training-data-repository-1"
BRANCH = "main"  # or 'master'
BATCHES = ["batch_1"]  # Later add batch_2, batch_3, etc.
MAX_SCORE = 15

# --- HELPERS ---
def github_url(batch, file):
    path=f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/{batch}/{file}"
    print(path)
    return path
def load_excel(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_excel(BytesIO(response.content))
    else:
        return None

def clean_attendance(df):
    df.columns = ['S.No', 'Name'] + df.iloc[0, 2:].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    df.iloc[:, 2:] = df.iloc[:, 2:].fillna('-')
    df['Days Present'] = df.iloc[:, 2:].apply(lambda row: (row == 'p').sum(), axis=1)
    total_days = len(df.columns[2:-2])
    df['Total Days'] = total_days
    df['Attendance %'] = (df['Days Present'] / total_days * 100).round(2)
    return df[['Name', 'Days Present', 'Attendance %']]

def clean_pretest(df):
    df = df[['Full name', 'Total points']].dropna()
    df.columns = ['Name', 'Score']
    df['Score %'] = (df['Score'] / MAX_SCORE * 100).round(2)
    return df

# --- UI: Select Batch ---
selected_batch = st.selectbox("📁 Select Batch", BATCHES)

# --- Load Data from GitHub ---
att_url = github_url(selected_batch, "attendance.xlsx")
pre_url = github_url(selected_batch, "pretest.xlsx")

att_df = load_excel(att_url)
pre_df = load_excel(pre_url)

if att_df is not None and pre_df is not None:
    # --- Process Data ---
    att_clean = clean_attendance(att_df)
    pre_clean = clean_pretest(pre_df)
    summary_df = pd.merge(att_clean, pre_clean, on="Name", how="outer")
    summary_df = summary_df.dropna(subset=["Name"])  # Drop rows with missing names

    # --- Display Table ---
    st.subheader(f"📋 Participant Summary for {selected_batch}")
    st.dataframe(summary_df, use_container_width=True)

    # --- Chart: Pre-Test Scores ---
    st.subheader("📈 Pre-Test Scores")
    chart_df = summary_df.dropna(subset=["Score"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(chart_df["Name"], chart_df["Score"], color="skyblue")
    ax.set_ylim(0, MAX_SCORE)
    ax.set_ylabel("Score")
    ax.set_title("Pre-Test Scores (out of 15)")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

else:
    st.warning("Failed to load one or both Excel files. Check file names or GitHub URLs.")
