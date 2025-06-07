import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests

st.set_page_config(page_title="Training Roll Out Dashboard", layout="wide")
st.title("🚀 Training Roll Out Dashboard")

# --- GitHub REPO DETAILS ---
GITHUB_USER = "NJ65-lang"
GITHUB_REPO = "training-data-repository-1"
BRANCH = "main"
BATCHES = ["batch_1", "batch_2"]  # Add more batch folders as needed
MAX_SCORE = 15

def github_url(batch, file):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/{batch}/{file}"

def load_excel(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_excel(BytesIO(response.content))
    else:
        return None

# --- BATCH SELECTOR ---
selected_batch = st.selectbox("📁 Select Training Batch", BATCHES)

# --- Load All Data from GitHub ---
att_url = github_url(selected_batch, "attendance.xlsx")
pre_url = github_url(selected_batch, "pretest.xlsx")
post_url = github_url(selected_batch, "posttest.xlsx")
fb_url = github_url(selected_batch, "feedback.xlsx")

attendance = load_excel(att_url)
pretest = load_excel(pre_url)
posttest = load_excel(post_url)
feedback = load_excel(fb_url)

if attendance is not None and pretest is not None and posttest is not None and feedback is not None:
    # --- Clean Attendance ---
    att_df = attendance.rename(columns={"Please enter your Name": "Name"})
    att_df['Name'] = att_df['Name'].astype(str).str.strip().str.lower()
    att_clean = att_df[['Name'] + [c for c in att_df.columns if c != 'Name']]

    # --- Clean Pre/Post Test ---
    def clean_test(df):
        df = df.rename(columns={"Full name": "Name"})
        df['Name'] = df['Name'].astype(str).str.strip().str.lower()
        df = df[['Name', 'Total points']].dropna(subset=['Name'])
        df['Total points'] = pd.to_numeric(df['Total points'], errors='coerce')
        return df

    pre_clean = clean_test(pretest)
    post_clean = clean_test(posttest)

    # --- Merge Pre/Post ---
    all_names = pd.DataFrame({'Name': pd.concat([att_clean['Name'], pre_clean['Name'], post_clean['Name']]).drop_duplicates()})
    scores_df = all_names \
        .merge(pre_clean.rename(columns={'Total points': 'Pretest'}), on='Name', how='left') \
        .merge(post_clean.rename(columns={'Total points': 'Posttest'}), on='Name', how='left')

    def score_display(x):
        return int(x) if pd.notnull(x) else "Not Appeared"

    scores_df['Pretest_Display'] = scores_df['Pretest'].apply(score_display)
    scores_df['Posttest_Display'] = scores_df['Posttest'].apply(score_display)
    scores_df['Improvement'] = scores_df.apply(
        lambda row: row['Posttest'] - row['Pretest'] if pd.notnull(row['Pretest']) and pd.notnull(row['Posttest']) else "Not Appeared",
        axis=1
    )
    scores_df['Improvement %'] = scores_df.apply(
        lambda row: round(row['Improvement'] / MAX_SCORE * 100, 2) if isinstance(row['Improvement'], (int, float, np.floating)) else "Not Appeared",
        axis=1
    )

    # --- Summary (only for both pre and post) ---
    valid_rows = scores_df[scores_df['Improvement'] != "Not Appeared"]
    total_participants = len(all_names)
    avg_pre = valid_rows['Pretest'].mean().round(2) if not valid_rows.empty else 0
    avg_post = valid_rows['Posttest'].mean().round(2) if not valid_rows.empty else 0
    avg_improve = valid_rows['Improvement'].mean().round(2) if not valid_rows.empty else 0

    # --- Feedback Cleaning ---
    fb = feedback.rename(columns={"Please enter your Name": "Name"})
    fb['Name'] = fb['Name'].astype(str).str.strip().str.lower()
    fb = fb[['Name', '1) Quality of Demos', '2) Presentation skills of trainer',
             '3) Query solving by trainer', '4) Overall quality of program']].copy()
    fb_scores = fb.set_index('Name').mean(axis=1)
    fb['Feedback Avg'] = fb_scores

    # --- UI Display ---
    st.header("📋 Attendance")
    st.dataframe(att_clean, use_container_width=True)

    st.header("📝 Pre/Post Test & Improvement")
    st.dataframe(scores_df[['Name', 'Pretest_Display', 'Posttest_Display', 'Improvement', 'Improvement %']], use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Participants", total_participants)
    c2.metric("📊 Avg Pre-Test (Both)", f"{avg_pre} / {MAX_SCORE}")
    c3.metric("📈 Avg Post-Test (Both)", f"{avg_post} / {MAX_SCORE}")
    c4.metric("🚀 Avg Improvement", f"{avg_improve}")

    if not valid_rows.empty:
        chart_df = valid_rows.sort_values(by="Improvement", ascending=False)
        fig, ax = plt.subplots(figsize=(max(10, len(chart_df) * 0.4), 6))
        bars = ax.bar(chart_df["Name"], chart_df["Improvement"], color="skyblue")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel("Improvement")
        ax.set_title("Improvement: Post-Test vs Pre-Test (Only Both)")
        plt.xticks(rotation=40, ha='right', fontsize=10)
        st.pyplot(fig)
    else:
        st.info("No participants appeared for both pre and post tests.")

    st.header("💬 Feedback (Averaged)")
    st.dataframe(fb[['Name', '1) Quality of Demos', '2) Presentation skills of trainer',
                     '3) Query solving by trainer', '4) Overall quality of program', 'Feedback Avg']], use_container_width=True)
    if not fb['Feedback Avg'].isnull().all():
        st.success(f"⭐ **Average Feedback Score:** {fb['Feedback Avg'].mean().round(2)}")
    else:
        st.info("No feedback scores available for this batch.")

else:
    st.error("❌ Failed to load one or more Excel files from GitHub. Please check your file names or batch folder in the repository.")
