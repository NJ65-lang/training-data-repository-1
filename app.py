import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests

st.set_page_config(page_title="Training Roll Out Dashboard", layout="wide")
st.markdown("<h1 style='font-size: 2.8rem;'>🚀 Training Roll Out Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .big-font { font-size: 22px !important; }
    </style>
""", unsafe_allow_html=True)

GITHUB_USER = "NJ65-lang"
GITHUB_REPO = "training-data-repository-1"
BRANCH = "main"
BATCHES = ["batch_1", "batch_2"]
MAX_SCORE = 15

def github_url(batch, file):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/{batch}/{file}"

def load_excel(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_excel(BytesIO(response.content))
    else:
        return None

def find_name_col(cols):
    for col in cols:
        if "name" in col.lower():
            return col
    return None

def clean_attendance(df):
    att_cols = df.columns.tolist()
    name_col = find_name_col(att_cols)
    if not name_col:
        st.error(f"Could not find a 'Name' column in attendance file. Columns found: {att_cols}")
        return None
    att_df = df.rename(columns={name_col: "Name"})
    att_df['Name'] = att_df['Name'].astype(str).str.strip().str.lower()
    keep_cols = ['Name'] + [c for c in att_df.columns if c != 'Name']
    return att_df[keep_cols]

def clean_test(df):
    test_cols = df.columns.tolist()
    name_col = find_name_col(test_cols)
    if not name_col or 'Total points' not in test_cols:
        st.error(f"Could not find a 'Name' or 'Total points' column in test file. Columns found: {test_cols}")
        return None
    test_df = df.rename(columns={name_col: "Name"})
    test_df['Name'] = test_df['Name'].astype(str).str.strip().str.lower()
    test_df = test_df[['Name', 'Total points']].dropna(subset=['Name'])
    test_df['Total points'] = pd.to_numeric(test_df['Total points'], errors='coerce')
    return test_df

def clean_feedback(df):
    fb_cols = df.columns.tolist()
    name_col = find_name_col(fb_cols)
    feedback_cols = [c for c in fb_cols if (c.strip().startswith('1)') or c.strip().startswith('2)') or c.strip().startswith('3)') or c.strip().startswith('4)'))]
    appreciations_col = None
    for c in fb_cols:
        if 'appreciation' in c.lower():
            appreciations_col = c
            break
    if not name_col or not feedback_cols:
        st.error(f"Could not find a 'Name' column or feedback questions in feedback file. Columns found: {fb_cols}")
        return None, None, None
    fb_df = df.rename(columns={name_col: "Name"})
    fb_df['Name'] = fb_df['Name'].astype(str).str.strip().str.lower()
    fb = fb_df[['Name'] + feedback_cols + ([appreciations_col] if appreciations_col else [])].copy()
    for col in feedback_cols:
        fb[col] = pd.to_numeric(fb[col], errors='coerce')
    return fb, feedback_cols, appreciations_col

selected_batch = st.selectbox("📁 Select Training Batch", BATCHES)

att_url = github_url(selected_batch, "attendance.xlsx")
pre_url = github_url(selected_batch, "pretest.xlsx")
post_url = github_url(selected_batch, "posttest.xlsx")
fb_url = github_url(selected_batch, "feedback.xlsx")

attendance = load_excel(att_url)
pretest = load_excel(pre_url)
posttest = load_excel(post_url)
feedback = load_excel(fb_url)

if attendance is not None and pretest is not None and posttest is not None and feedback is not None:
    att_clean = clean_attendance(attendance)
    pre_clean = clean_test(pretest)
    post_clean = clean_test(posttest)
    fb, feedback_cols, appreciations_col = clean_feedback(feedback)

    if any(x is None for x in [att_clean, pre_clean, post_clean, fb]):
        st.error("One or more files could not be parsed due to missing key columns. See above errors.")
    else:
        # Attendance Table
        st.markdown("<h2 class='big-font'>📋 Attendance</h2>", unsafe_allow_html=True)
        st.markdown(att_clean.to_html(index=False, classes='big-font'), unsafe_allow_html=True)

        # Pre-test Table
        st.markdown("<h2 class='big-font'>🧪 Pre-Test Scores</h2>", unsafe_allow_html=True)
        pre_disp = pre_clean.copy()
        pre_disp['Total points'] = pre_disp['Total points'].apply(lambda x: int(x) if pd.notnull(x) else "NA")
        pre_disp['Name'] = pre_disp['Name'].str.title()   # Capitalize name for display
        st.markdown(pre_disp.to_html(index=False, classes='big-font'), unsafe_allow_html=True)

        # Post-test Table
        st.markdown("<h2 class='big-font'>🏁 Post-Test Scores</h2>", unsafe_allow_html=True)
        post_disp = post_clean.copy()
        post_disp['Total points'] = post_disp['Total points'].apply(lambda x: int(x) if pd.notnull(x) else "NA")
        post_disp['Name'] = post_disp['Name'].str.title()
        st.markdown(post_disp.to_html(index=False, classes='big-font'), unsafe_allow_html=True)

        # Feedback Chart (improved x-axis readability)
        st.markdown("<h2 style='font-size:1.8rem;'>💬 Feedback Summary (Parameter-wise Averages)</h2>", unsafe_allow_html=True)
        feedback_avgs = fb[feedback_cols].mean().round(2)
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        bars2 = ax2.bar(feedback_avgs.index, feedback_avgs.values, color='#6C63FF')
        ax2.set_ylim(0, 5)
        for i, bar in enumerate(bars2):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f'{yval:.2f}', ha='center', va='bottom', fontsize=14)
        ax2.set_title("Average Feedback per Parameter", fontsize=17)
        ax2.set_ylabel("Average Score (out of 5)", fontsize=13)
        # Make X labels not overlap
        ax2.set_xticks(np.arange(len(feedback_avgs.index)))
        ax2.set_xticklabels(feedback_avgs.index, rotation=20, ha='right', fontsize=14, wrap=True)
        ax2.tick_params(axis='y', labelsize=12)
        fig2.tight_layout()
        st.pyplot(fig2)

        # Appreciations Section
        if appreciations_col:
            appreciations = fb[appreciations_col].dropna().astype(str).unique()
            if appreciations.size > 0:
                st.markdown("<h2 style='font-size:1.6rem;'>🌟 Appreciations</h2>", unsafe_allow_html=True)
                for app in appreciations:
                    st.markdown(
                        f"<div style='margin-bottom:18px; font-size:1.4rem; color:#4B3F72; font-style:italic; font-family:serif;'>"
                        f"“{app.strip()}”</div>",
                        unsafe_allow_html=True)
            else:
                st.info("No appreciations/comments were provided.")
        else:
            st.info("No appreciations/comments column found in feedback.")

else:
    st.error("❌ Failed to load one or more Excel files from GitHub. Please check your file names or batch folder in the repository.")
