# merge_reviews.py
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

# ==========================================
# 1️⃣ CONFIGURE GEMINI
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==========================================
# 2️⃣ LOAD CSV FILES
# ==========================================
reviews_path = "customer_reviews.csv"
shipping_path = "shipping_logs.csv"

if not os.path.exists(reviews_path) or not os.path.exists(shipping_path):
    raise FileNotFoundError("❌ Missing one of the input files: customer_reviews.csv or shipping_logs.csv")

reviews_df = pd.read_csv(reviews_path)
shipping_df = pd.read_csv(shipping_path)

print("✅ Loaded:")
print(f" - {reviews_path} → {len(reviews_df)} rows")
print(f" - {shipping_path} → {len(shipping_df)} rows")

# ==========================================
# 3️⃣ MERGE TABLES ON 'Order ID'
# ==========================================
reviews_df.columns = reviews_df.columns.str.strip()
shipping_df.columns = shipping_df.columns.str.strip()

merged_df = pd.merge(reviews_df, shipping_df, on="Order ID", how="inner")
print(f"✅ Merged dataset: {len(merged_df)} rows")

# ==========================================
# 4️⃣ CLEAN + RENAME COLUMNS FOR CONSISTENCY
# ==========================================
merged_df.columns = (
    merged_df.columns.str.lower().str.strip().str.replace(" ", "_")
)

# Rename columns for dashboard compatibility
merged_df = merged_df.rename(columns={
    "summary": "review_text",
    "product": "product",
    "region": "region",
    "status": "status",
    "sentiment_score": "sentiment_score"
})

# Fill missing columns if not present
for col in ["region", "status", "sentiment_score"]:
    if col not in merged_df.columns:
        merged_df[col] = "Unknown" if col != "sentiment_score" else 0.0

# ==========================================
# 5️⃣ GENERATE SENTIMENT SCORES (IF MISSING)
# ==========================================
def get_sentiment_score(text):
    """Rate sentiment using Gemini (-1 = negative, +1 = positive)."""
    if not text or pd.isna(text):
        return 0.0
    try:
        prompt = f"Rate the sentiment of this customer review on a scale from -1 (negative) to +1 (positive). Only return a number.\nReview: {text}"
        response = model.generate_content(prompt)
        result = response.text.strip()
        return float(result)
    except Exception:
        return 0.0

if (merged_df["sentiment_score"] == 0).all():
    print("🔄 Generating sentiment scores using Gemini...")
    tqdm.pandas(desc="Analyzing sentiment")
    merged_df["sentiment_score"] = merged_df["review_text"].progress_apply(get_sentiment_score)
    print("✅ Sentiment analysis completed!")

# ==========================================
# 6️⃣ VISUALIZE SENTIMENT DISTRIBUTION
# ==========================================
plt.figure(figsize=(8, 6))
plt.hist(merged_df["sentiment_score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram of Sentiment Scores (Gemini)")
plt.xlabel("Sentiment Score (-1 to +1)")
plt.ylabel("Number of Reviews")
plt.grid(axis="y", alpha=0.7)
plt.show()

# ==========================================
# 7️⃣ SAVE CLEAN DATASET
# ==========================================
output_path = "reviews_with_sentiment.csv"
merged_df.to_csv(output_path, index=False)
print(f"✅ Saved clean merged dataset as '{output_path}'")

# Show sample output
print("\n📊 Preview of Clean Dataset:")
print(merged_df.head())
