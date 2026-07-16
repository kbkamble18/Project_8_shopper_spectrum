import streamlit as st
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_artifacts():
    return (
        joblib.load("artifacts/kmeans_model.pkl"),
        joblib.load("artifacts/rfm_scaler.pkl"),
        joblib.load("artifacts/segment_mapping.pkl"),
        joblib.load("artifacts/item_similarity.pkl"),
        joblib.load("artifacts/item_lookup.pkl"),
        joblib.load("artifacts/summary_stats.pkl"),
        joblib.load("artifacts/top_countries.pkl"),
        joblib.load("artifacts/top_products.pkl"),
        joblib.load("artifacts/monthly_trend.pkl"),
        joblib.load("artifacts/rfm_segmented.pkl"),
        joblib.load("artifacts/popular_products.pkl"),
    )


(
    kmeans,
    scaler,
    segment_mapping,
    item_sim_df,
    item_lookup,
    summary,
    top_countries,
    top_products,
    monthly_trend,
    rfm,
    popular_products,
) = load_artifacts()

# ====================== SIDEBAR ======================
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Dashboard",
        "RFM Analysis",
        "Elbow Method",
        "Customer Segmentation",
        "Product Recommendations",
        "Predict Customer Segment",
        "Business Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("RFM + KMeans + Collaborative Filtering")

# ====================== PAGES ======================

if page == "Executive Dashboard":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{summary['total_customers']:,}")
    col2.metric("Total Revenue", f"${summary['total_revenue']:,.0f}")
    col3.metric("Total Transactions", f"{summary['total_transactions']:,}")
    col4.metric("Silhouette Score", "0.616")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Revenue by Country (Top 10)")
        fig = px.bar(top_countries, x="Revenue", y="Country", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Monthly Revenue Trend")
        fig2 = px.line(monthly_trend, x="InvoiceDate", y="Revenue", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Products by Revenue")
    fig3 = px.bar(top_products, x="TotalAmount", y="Description", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)


elif page == "RFM Analysis":
    st.title("📊 RFM Analysis")
    st.subheader("Distribution of Recency, Frequency & Monetary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            px.histogram(rfm, x="Recency", nbins=50, title="Recency"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.histogram(rfm, x="Frequency", nbins=50, title="Frequency"),
            use_container_width=True,
        )
    with col3:
        st.plotly_chart(
            px.histogram(rfm, x="Monetary", nbins=50, title="Monetary"),
            use_container_width=True,
        )

    st.subheader("RFM Summary Statistics")
    st.dataframe(
        rfm[["Recency", "Frequency", "Monetary"]].describe().round(2),
        use_container_width=True,
    )


elif page == "Elbow Method":
    st.title("📉 Elbow Method")
    st.subheader("Optimal Number of Clusters")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    rfm_scaled = StandardScaler().fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    inertias = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(rfm_scaled).inertia_
        for k in range(2, 11)
    ]

    fig = px.line(x=list(range(2, 11)), y=inertias, markers=True, title="Elbow Curve")
    fig.update_layout(
        xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Optimal number of clusters selected: **k=4** (based on Elbow Method)")


elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")

    st.subheader("2D View: Recency vs Monetary")
    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Segment",
        size="Frequency",
        hover_data=["Segment"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3D View: RFM Clusters")
    fig3d = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary", color="Segment"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("Segment Distribution")
    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    st.dataframe(segment_counts, use_container_width=True)


elif page == "Product Recommendations":
    st.title("🎯 Product Recommendations")
    st.subheader("Item-Based Collaborative Filtering")

    selected_product = st.selectbox("Select a Product", options=popular_products)

    if st.button("Get Recommendations", type="primary"):
        matches = item_lookup[
            item_lookup["Description"].str.contains(
                selected_product, case=False, na=False
            )
        ]
        if not matches.empty:
            stock_code = matches.iloc[0]["StockCode"]
            if stock_code in item_sim_df.index:
                sims = item_sim_df[stock_code].sort_values(ascending=False).iloc[1:6]
                rec_df = item_lookup[item_lookup["StockCode"].isin(sims.index)][
                    ["Description"]
                ].reset_index(drop=True)

                st.success(f"Top 5 recommendations for **{selected_product}**")
                for i, row in rec_df.iterrows():
                    st.markdown(f"**{i + 1}.** {row['Description']}")


elif page == "Predict Customer Segment":
    st.title("🔍 Predict Customer Segment")

    col1, col2, col3 = st.columns(3)
    recency = col1.number_input("Recency (days)", min_value=0, value=45)
    frequency = col2.number_input("Frequency", min_value=1, value=5)
    monetary = col3.number_input("Monetary ($)", min_value=0.0, value=1200.0)

    if st.button("Predict Segment", type="primary"):
        cluster = kmeans.predict(scaler.transform([[recency, frequency, monetary]]))[0]
        segment = segment_mapping.get(cluster, "Unknown")
        st.success(f"**Predicted Segment: {segment}**")


elif page == "Business Insights":
    st.title("💡 Business Insights")

    st.subheader("Key Findings")
    st.markdown("""
    - United Kingdom dominates revenue.
    - High-Value + VIP customers (~5%) drive disproportionate revenue.
    - ~25% of customers are At-Risk and need immediate attention.
    """)

    st.subheader("Segment Distribution")
    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    segment_counts["Percentage"] = (
        segment_counts["Count"] / segment_counts["Count"].sum() * 100
    ).round(1)
    st.dataframe(segment_counts, use_container_width=True)

    st.subheader("Recommended Actions")
    st.markdown("""
    - Launch loyalty programs for **High-Value** and **VIP** segments.
    - Run reactivation campaigns for **At-Risk** customers.
    - Use product recommendations in marketing automation.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python • Scikit-learn • Streamlit")

# new uodate