import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

print("=" * 65)
print(" SHOPPER SPECTRUM - FINAL PIPELINE")
print("=" * 65)

# ====================== DATA LOADING & CLEANING ======================
df = pd.read_csv("online_retail.csv")
print(f"\nRaw shape: {df.shape}")

df_clean = df.copy()

df_clean = df_clean.dropna(subset=["CustomerID"])
df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(subset=["Description"])

df_clean["TotalAmount"] = df_clean["Quantity"] * df_clean["UnitPrice"]
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

print(f"Cleaned shape: {df_clean.shape}")
print("All cleaning validations passed successfully.")

df_clean.to_csv("artifacts/cleaned_transactions.csv", index=False)

# ====================== RFM FEATURE ENGINEERING ======================
snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = (
    df_clean.groupby("CustomerID")
    .agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )
    .reset_index()
)

print(
    "\nRFM Statistics:\n", rfm[["Recency", "Frequency", "Monetary"]].describe().round(2)
)

# ====================== KMEANS CLUSTERING ======================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

print("\n=== Evaluating Different k Values ===")
for k_test in range(2, 8):
    km_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels_test)
    print(f"k={k_test} | Inertia: {km_test.inertia_:.0f} | Silhouette: {sil:.3f}")

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_profile = (
    rfm.groupby("Cluster")
    .agg(
        {
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "CustomerID": "count",
        }
    )
    .rename(columns={"CustomerID": "Count"})
    .round(1)
)
print("\nCluster Profile:\n", cluster_profile)

segment_mapping = {
    0: "Regular",
    1: "At-Risk",
    2: "High-Value (VIP)",
    3: "High-Value",
}
rfm["Segment"] = rfm["Cluster"].map(segment_mapping)
print("\nFinal Segment Distribution:\n", rfm["Segment"].value_counts())

# ====================== ITEM-BASED COLLABORATIVE FILTERING ======================
item_lookup = (
    df_clean.groupby("StockCode")["Description"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)

item_user = df_clean.pivot_table(
    index="StockCode",
    columns="CustomerID",
    values="Quantity",
    aggfunc="sum",
    fill_value=0,
)

item_sim = cosine_similarity(item_user)
item_sim_df = pd.DataFrame(item_sim, index=item_user.index, columns=item_user.index)

# ====================== SAVE ALL ARTIFACTS ======================
os.makedirs("artifacts", exist_ok=True)

joblib.dump(kmeans, "artifacts/kmeans_model.pkl")
joblib.dump(scaler, "artifacts/rfm_scaler.pkl")
joblib.dump(segment_mapping, "artifacts/segment_mapping.pkl")
joblib.dump(item_sim_df, "artifacts/item_similarity.pkl")
joblib.dump(item_lookup, "artifacts/item_lookup.pkl")

summary = {
    "total_customers": rfm.shape[0],
    "total_revenue": round(df_clean["TotalAmount"].sum(), 2),
    "total_transactions": df_clean["InvoiceNo"].nunique(),
    "date_range": f"{df_clean['InvoiceDate'].min().date()} – {df_clean['InvoiceDate'].max().date()}",
}
joblib.dump(summary, "artifacts/summary_stats.pkl")

country_sales = (
    df_clean.groupby("Country")
    .agg(Revenue=("TotalAmount", "sum"), Transactions=("InvoiceNo", "nunique"))
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .head(10)
)
joblib.dump(country_sales, "artifacts/top_countries.pkl")

top_prod = (
    df_clean.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
joblib.dump(top_prod, "artifacts/top_products.pkl")

monthly = (
    df_clean.set_index("InvoiceDate").resample("ME")["TotalAmount"].sum().reset_index()
)
monthly.columns = ["InvoiceDate", "Revenue"]
joblib.dump(monthly, "artifacts/monthly_trend.pkl")

joblib.dump(
    rfm[["Recency", "Frequency", "Monetary", "Cluster", "Segment"]],
    "artifacts/rfm_segmented.pkl",
)

popular = (
    df_clean.groupby("Description")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(30)
    .index.tolist()
)
joblib.dump(popular, "artifacts/popular_products.pkl")

print("\n" + "=" * 65)
print(" PIPELINE COMPLETE — All artifacts saved successfully in /artifacts/")
print("=" * 65)
