import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ============================================
# 1️⃣ CONFIGURE GEMINI
# ============================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found! Add it to your .env file.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ============================================
# 2️⃣ LOAD DATA LOCALLY
# ============================================
st.title("📊 Product Intelligence Dashboard (Local + Gemini)")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.upper()

    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    # Product Filter
    if "PRODUCT" in df.columns:
        products = df["PRODUCT"].unique()
        selected_products = st.multiselect("Select Products:", products, default=list(products))
        filtered_df = df[df["PRODUCT"].isin(selected_products)]
    else:
        st.warning("⚠️ 'PRODUCT' column not found. Skipping product filter.")
        filtered_df = df

    # ============================================
    # SENTIMENT VISUALIZATION
    # ============================================
    if {"SENTIMENT_SCORE"}.issubset(filtered_df.columns):
        st.subheader("💬 Average Sentiment by Product")
        try:
            product_sentiment = filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().sort_values()
            fig, ax = plt.subplots()
            product_sentiment.plot(kind="barh", ax=ax)
            ax.set_title("Average Sentiment by Product")
            ax.set_xlabel("Sentiment Score")
            st.pyplot(fig)
        except:
            st.warning("⚠️ Unable to calculate sentiment scores.")
    else:
        st.warning("⚠️ 'SENTIMENT_SCORE' column missing. Cannot show sentiment chart.")

    # ============================================
    # CHATBOT
    # ============================================
    st.subheader("🤖 Ask Questions About Your Data")
    user_question = st.text_input("Enter your question here:")

    if user_question:
        with st.spinner("Thinking..."):
            df_string = filtered_df.to_string(index=False)
            prompt = f"""
            You are a data analyst. Answer using ONLY the dataset below.

            DATA:
            {df_string}

            QUESTION:
            {user_question}
            """

            try:
                response = model.generate_content([prompt])
                st.write(response.text)
            except Exception as e:
                st.error(f"⚠️ Gemini API Error: {e}")

else:
    st.info("📤 Upload a CSV or Excel file to begin.")
