"""
Zomato Restaurant Clustering & Sentiment Analysis
Streamlit Deployment App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Restaurant Clustering",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #e23744 0%, #c0392b 100%);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1.2rem;
        border-left: 4px solid #e23744; margin-bottom: 1rem;
    }
    .cluster-card {
        border-radius: 10px; padding: 1.2rem; margin-bottom: 0.8rem;
        color: white; font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 20px;
    }
    .insight-box {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Cluster metadata ──────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: ("💎 Premium Fine Dining",  "#8e44ad"),
    1: ("🍕 Popular Casual",       "#2980b9"),
    2: ("💰 Budget Gems",          "#27ae60"),
    3: ("⚠️ Underperformers",      "#e67e22"),
}
CLUSTER_COLORS = ["#8e44ad", "#2980b9", "#27ae60", "#e67e22"]

# ── Load models and data ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler        = joblib.load('scaler.joblib')
    kmeans        = joblib.load('kmeans_final.joblib')
    feature_cols  = joblib.load('feature_cols.joblib')
    top10_cuisines= joblib.load('top10_cuisines.joblib')
    return scaler, kmeans, feature_cols, top10_cuisines

@st.cache_data
def load_data():
    df_feat    = pd.read_csv('df_feat.csv')
    df_meta    = pd.read_csv('df_meta.csv')
    df_reviews = pd.read_csv('df_reviews.csv')
    return df_feat, df_meta, df_reviews

try:
    scaler, kmeans, feature_cols, top10_cuisines = load_models()
    df_feat, df_meta, df_reviews = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── VADER sentiment (lightweight, no model file needed) ───────────────────────
@st.cache_resource
def load_vader():
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

sid = load_vader()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png",
             width=160)
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", [
        "🏠 Home & Overview",
        "📊 EDA Dashboard",
        "🤖 Predict Restaurant Cluster",
        "💬 Review Sentiment Analyser",
        "📈 Model Performance",
        "🗂️ Restaurant Explorer",
    ])
    st.markdown("---")
    st.markdown("**Project:** Unsupervised ML + NLP")
    st.markdown("**Models:** K-Means · VADER")
    st.markdown("**Dataset:** 105 restaurants · 10K reviews")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home & Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Zomato Restaurant Intelligence Platform</h1>
        <p style="font-size:1.1rem;">Unsupervised ML · Sentiment Analysis · Restaurant Clustering</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"⚠️ Models not found. Place model files in the same directory as app.py.\nError: {load_error}")
        st.stop()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏪 Restaurants", f"{len(df_feat):,}")
    col2.metric("📝 Reviews", f"{len(df_reviews):,}")
    col3.metric("🍴 Cuisines", f"{df_meta['Cuisines'].str.split(', ').explode().nunique()}")
    col4.metric("🔢 Clusters", "4")

    st.markdown("---")

    # Cluster summary cards
    st.subheader("🔍 Restaurant Segments Identified")
    cluster_descriptions = {
        0: ("High cost, high rating, selective audience.",
            "Target for premium ad placements & corporate dining partnerships."),
        1: ("Highest review volume, good value, Zomato's revenue backbone.",
            "Prioritise listing quality & exclusive discount campaigns."),
        2: ("Low cost, decent quality, underserved marketing segment.",
            "Promote via 'Best Value' collections & student deals."),
        3: ("Below-average ratings & sentiment, need intervention.",
            "Flag for Zomato restaurant success team outreach."),
    }

    cols = st.columns(2)
    for k, (label, color) in CLUSTER_LABELS.items():
        count = (df_feat['KMeans_Cluster'] == k).sum()
        desc, action = cluster_descriptions[k]
        with cols[k % 2]:
            st.markdown(f"""
            <div class="cluster-card" style="background:{color};">
                <div style="font-size:1.1rem;">{label} &nbsp;|&nbsp; {count} restaurants</div>
                <div style="font-weight:normal; font-size:0.85rem; margin-top:6px;">{desc}</div>
                <div style="font-weight:normal; font-size:0.82rem; margin-top:4px;
                            opacity:0.9;">💡 {action}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 Key Business Insights")
    insights = [
        ("💰", "Cost Distribution", "Most restaurants cluster in ₹500–₹1,500 (mid-segment). Premium tier (>₹2,000) is a small niche."),
        ("⭐", "Rating Bias", "38% of reviews are 5-star — significant positive rating bias platform-wide."),
        ("📅", "Weekend Effect", "Saturday & Sunday drive 40% more reviews than weekdays — dining is a leisure activity."),
        ("😊", "Sentiment vs Rating", "~20% of 3-star reviews carry negative VADER sentiment — 'polite dissatisfaction' hidden in ratings."),
        ("🕵️", "Critic Influence", "A small cohort of 50+ review power-users drives disproportionate restaurant reputation."),
        ("🍛", "Top Cuisines", "North Indian & Chinese dominate. Continental & Italian command the highest average cost."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(insights):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{icon}</div>
                <strong>{title}</strong>
                <div style="font-size:0.85rem; color:#555; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    if not models_loaded:
        st.error("Models not loaded."); st.stop()

    tab1, tab2, tab3 = st.tabs(["💰 Cost & Cuisine", "⭐ Ratings & Sentiment", "📅 Temporal & Engagement"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cost Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_meta['Cost_INR'].dropna(), bins=15, color='#e23744',
                    edgecolor='white', alpha=0.85)
            ax.axvline(df_meta['Cost_INR'].median(), color='navy',
                       linestyle='--', label=f"Median ₹{df_meta['Cost_INR'].median():.0f}")
            ax.set_xlabel('Cost for Two (INR)'); ax.set_ylabel('Restaurants')
            ax.set_title('Restaurant Cost Distribution')
            ax.legend(fontsize=9); plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Top 12 Cuisines")
            all_c = pd.Series([c.strip() for lst in
                               df_meta['Cuisine_List'].apply(
                                   lambda x: eval(x) if isinstance(x, str) else x)
                               for c in lst if c.strip()])
            top12 = all_c.value_counts().head(12)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top12.index[::-1], top12.values[::-1],
                    color=sns.color_palette('Reds_r', 12))
            ax.set_title('Top 12 Cuisines')
            ax.set_xlabel('Number of Restaurants')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Cuisine vs Cost
        st.subheader("Average Cost by Cuisine")
        cuisine_cost_rows = []
        for _, row in df_meta.iterrows():
            cuisine_list = eval(row['Cuisine_List']) if isinstance(row['Cuisine_List'], str) else row['Cuisine_List']
            for c in cuisine_list:
                c = c.strip()
                if c:
                    cuisine_cost_rows.append({'Cuisine': c, 'Cost_INR': row['Cost_INR']})
        df_cc = pd.DataFrame(cuisine_cost_rows).dropna()
        top10_c = all_c.value_counts().head(10).index.tolist()
        avg_cc = (df_cc[df_cc['Cuisine'].isin(top10_c)]
                  .groupby('Cuisine')['Cost_INR'].mean()
                  .sort_values(ascending=False).reset_index())
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=avg_cc, x='Cost_INR', y='Cuisine', ax=ax, palette='coolwarm')
        ax.set_title('Average Cost for Two by Cuisine')
        ax.set_xlabel('Avg Cost (INR)')
        for p in ax.patches:
            ax.text(p.get_width()+10, p.get_y()+p.get_height()/2,
                    f'₹{p.get_width():.0f}', va='center', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rating Distribution")
            ratings = df_reviews['Rating_Clean'].dropna()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ratings, bins=10, color='coral', edgecolor='white')
            ax.axvline(ratings.mean(), color='navy', linestyle='--',
                       label=f'Mean {ratings.mean():.2f}')
            ax.set_xlabel('Rating'); ax.set_ylabel('Count')
            ax.set_title('Review Rating Distribution')
            ax.legend(fontsize=9); plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Sentiment Distribution")
            if 'Sentiment_Label' in df_reviews.columns:
                sl = df_reviews['Sentiment_Label'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(sl, labels=sl.index, autopct='%1.1f%%',
                       colors=['#2ecc71','#e74c3c','#f39c12'],
                       startangle=140, wedgeprops=dict(edgecolor='white'))
                ax.set_title('Overall Sentiment Split')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                st.info("Sentiment column not found in loaded data.")

        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        num_cols = ['Cost_INR_capped','Avg_Rating','Review_Count',
                    'Avg_Sentiment','Pct_Positive','Avg_Pictures']
        avail = [c for c in num_cols if c in df_feat.columns]
        corr = df_feat[avail].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, linewidths=0.5, linecolor='white', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Reviews by Day of Week")
            if 'Review_DayOfWeek' in df_reviews.columns:
                day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
                day_counts = df_reviews['Review_DayOfWeek'].map(day_map).value_counts()
                day_counts = day_counts.reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
                colors = ['#e74c3c' if d in ['Sat','Sun'] else '#3498db'
                          for d in day_counts.index]
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(day_counts.index, day_counts.values,
                              color=colors, edgecolor='white')
                ax.set_title('Review Volume by Day')
                ax.set_ylabel('Reviews')
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+20,
                            f'{bar.get_height():.0f}',
                            ha='center', fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Pictures vs Rating")
            if 'Pictures' in df_reviews.columns:
                pic_r = (df_reviews.groupby(
                    df_reviews['Rating_Clean'].round())['Pictures']
                    .mean().reset_index())
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(pic_r['Rating_Clean'], pic_r['Pictures'],
                       color='steelblue', edgecolor='white')
                ax.set_title('Avg Pictures Posted by Rating')
                ax.set_xlabel('Rating'); ax.set_ylabel('Avg Pictures')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # Top reviewed restaurants
        st.subheader("Top 15 Most-Reviewed Restaurants")
        top_rev = df_reviews['Restaurant'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_rev.values, y=top_rev.index, ax=ax, palette='viridis')
        ax.set_xlabel('Number of Reviews')
        ax.set_title('Top 15 Most-Reviewed Restaurants')
        for i, v in enumerate(top_rev.values):
            ax.text(v+5, i, str(v), va='center', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT RESTAURANT CLUSTER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Predict Restaurant Cluster":
    st.title("🤖 Predict Restaurant Cluster")
    st.markdown("Enter restaurant details below to predict which business segment it belongs to.")
    if not models_loaded:
        st.error("Models not loaded."); st.stop()

    with st.form("predict_form"):
        st.subheader("📋 Restaurant Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            cost       = st.number_input("💰 Cost for Two (₹)", 100, 5000, 1200, step=50)
            avg_rating = st.slider("⭐ Average Rating", 1.0, 5.0, 4.0, step=0.1)
            num_reviews= st.number_input("📝 Number of Reviews", 1, 2000, 80)

        with col2:
            avg_sentiment  = st.slider("😊 Avg Sentiment Score", -1.0, 1.0, 0.3, step=0.05)
            pct_positive   = st.slider("✅ % Positive Reviews", 0, 100, 65) / 100
            avg_pictures   = st.number_input("📸 Avg Pictures per Review", 0.0, 10.0, 1.5, step=0.5)

        with col3:
            avg_followers    = st.number_input("👥 Avg Reviewer Followers", 0, 500, 3)
            avg_critic_score = st.number_input("🎖️ Avg Reviewer Review Count", 0, 500, 10)
            selected_cuisines= st.multiselect("🍴 Cuisines Offered",
                                              top10_cuisines,
                                              default=['North Indian'])

        submitted = st.form_submit_button("🔮 Predict Cluster", use_container_width=True)

    if submitted:
        # Build feature dict
        unseen = {
            'Cost_INR_capped'  : float(cost),
            'Avg_Rating'       : float(avg_rating),
            'log_Review_Count' : np.log1p(num_reviews),
            'Avg_Pictures'     : np.log1p(avg_pictures),
            'Avg_Followers'    : np.log1p(avg_followers),
            'Avg_Critic_Score' : np.log1p(avg_critic_score),
            'Avg_Sentiment'    : float(avg_sentiment),
            'Pct_Positive'     : float(pct_positive),
        }
        for cuisine in top10_cuisines:
            col_name = f"CuisineOH_{cuisine.replace(' ','_').replace('/','_')}"
            unseen[col_name] = 1 if cuisine in selected_cuisines else 0

        unseen_df = pd.DataFrame([unseen])
        for col in feature_cols:
            if col not in unseen_df.columns:
                unseen_df[col] = 0
        unseen_df     = unseen_df[feature_cols]
        unseen_scaled = scaler.transform(unseen_df)
        pred_cluster  = kmeans.predict(unseen_scaled)[0]
        distances     = cdist(unseen_scaled, kmeans.cluster_centers_, metric='euclidean')[0]
        confidence    = 1 - (distances[pred_cluster] / distances.sum())

        label, color = CLUSTER_LABELS[pred_cluster]
        st.markdown("---")
        st.markdown(f"""
        <div class="cluster-card" style="background:{color}; font-size:1.3rem; text-align:center;">
            Predicted Segment: {label}
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Assigned Cluster", f"Cluster {pred_cluster}")
        col2.metric("Segment", label.split(' ',1)[1])
        col3.metric("Confidence", f"{confidence*100:.1f}%")

        # Distance to all centroids
        st.subheader("📏 Distance to All Cluster Centroids")
        dist_df = pd.DataFrame({
            'Cluster': [f"C{k}: {CLUSTER_LABELS[k][0]}" for k in range(4)],
            'Distance': distances,
            'Assigned': ['✅ YES' if k == pred_cluster else '' for k in range(4)]
        })
        fig, ax = plt.subplots(figsize=(10, 3))
        bar_colors = [CLUSTER_COLORS[k] for k in range(4)]
        bars = ax.bar(dist_df['Cluster'], dist_df['Distance'],
                      color=bar_colors, edgecolor='white', alpha=0.85)
        ax.set_title('Euclidean Distance to Each Cluster Centroid (lower = closer)')
        ax.set_ylabel('Distance')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, distances):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Business recommendation
        recs = {
            0: "💡 List on Zomato's **Premium Dining** collection. Eligible for corporate tie-ups and exclusive event hosting.",
            1: "💡 Prioritise listing quality and photo count. Ideal candidate for **Zomato Gold** membership and deal campaigns.",
            2: "💡 Feature in **'Best Value'** and student-friendly collections. Push notifications on weekday lunch hours.",
            3: "💡 Flag for **Restaurant Success Team** review. Focus on service speed and food temperature improvements.",
        }
        st.info(recs[pred_cluster])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SENTIMENT ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 Review Sentiment Analyser":
    st.title("💬 Review Sentiment Analyser")
    st.markdown("Enter any restaurant review text to get an instant VADER sentiment analysis.")

    # Single review analyser
    st.subheader("🔍 Analyse a Single Review")
    review_text = st.text_area("Paste your review here:",
                               placeholder="e.g. The food was absolutely amazing, service was quick and staff very friendly!",
                               height=120)

    if st.button("Analyse Sentiment", use_container_width=True):
        if review_text.strip():
            scores = sid.polarity_scores(review_text)
            compound = scores['compound']

            if compound >= 0.05:
                label, color, emoji = "Positive", "#27ae60", "😊"
            elif compound <= -0.05:
                label, color, emoji = "Negative", "#e74c3c", "😞"
            else:
                label, color, emoji = "Neutral", "#f39c12", "😐"

            st.markdown(f"""
            <div class="cluster-card" style="background:{color}; text-align:center; font-size:1.2rem;">
                {emoji} Sentiment: {label} &nbsp;|&nbsp; Compound Score: {compound:.4f}
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compound", f"{compound:.4f}")
            col2.metric("Positive", f"{scores['pos']:.3f}")
            col3.metric("Neutral",  f"{scores['neu']:.3f}")
            col4.metric("Negative", f"{scores['neg']:.3f}")

            # Score bar
            fig, ax = plt.subplots(figsize=(8, 1.8))
            ax.barh(['neg','neu','pos'],
                    [scores['neg'], scores['neu'], scores['pos']],
                    color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white')
            ax.set_xlim(0, 1)
            ax.set_title('VADER Component Scores')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.warning("Please enter a review to analyse.")

    st.markdown("---")

    # Batch analyser from dataset
    st.subheader("📊 Live Sentiment Stats from Dataset")
    if not models_loaded:
        st.info("Load models to see dataset stats.")
    else:
        if 'Sentiment_Label' in df_reviews.columns:
            col1, col2, col3 = st.columns(3)
            sl = df_reviews['Sentiment_Label'].value_counts()
            col1.metric("😊 Positive", f"{sl.get('Positive',0):,}",
                        f"{sl.get('Positive',0)/len(df_reviews)*100:.1f}%")
            col2.metric("😐 Neutral",  f"{sl.get('Neutral',0):,}",
                        f"{sl.get('Neutral',0)/len(df_reviews)*100:.1f}%")
            col3.metric("😞 Negative", f"{sl.get('Negative',0):,}",
                        f"{sl.get('Negative',0)/len(df_reviews)*100:.1f}%")

            # Restaurant selector
            st.subheader("🏪 Sentiment Profile by Restaurant")
            restaurants = sorted(df_reviews['Restaurant'].unique())
            selected_r  = st.selectbox("Select a restaurant:", restaurants)
            r_data = df_reviews[df_reviews['Restaurant'] == selected_r]

            if not r_data.empty:
                r_sl = r_data['Sentiment_Label'].value_counts()
                avg_s = r_data['Sentiment_Score'].mean()
                avg_rt = r_data['Rating_Clean'].mean()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Reviews", f"{len(r_data):,}")
                col2.metric("Avg Rating", f"{avg_rt:.2f} ⭐")
                col3.metric("Avg Sentiment", f"{avg_s:.3f}")
                col4.metric("Top Sentiment", r_sl.index[0] if len(r_sl) else "N/A")

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                axes[0].pie(r_sl, labels=r_sl.index, autopct='%1.1f%%',
                            colors=['#2ecc71','#e74c3c','#f39c12'],
                            startangle=140, wedgeprops=dict(edgecolor='white'))
                axes[0].set_title(f'Sentiment – {selected_r[:25]}')

                if 'Review_Month' in r_data.columns:
                    monthly = r_data.groupby('Review_Month')['Sentiment_Score'].mean()
                    axes[1].plot(monthly.index, monthly.values, 'o-',
                                 color='#e23744', linewidth=2)
                    axes[1].set_title('Avg Sentiment by Month')
                    axes[1].set_xlabel('Month'); axes[1].set_ylabel('Sentiment Score')
                    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)

                plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    if not models_loaded:
        st.error("Models not loaded."); st.stop()

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler

    feature_cols_avail = [c for c in feature_cols if c in df_feat.columns]
    df_feat_num = df_feat[feature_cols_avail].fillna(df_feat[feature_cols_avail].median())
    X_sc = scaler.transform(df_feat_num)
    labels = kmeans.labels_

    sil  = silhouette_score(X_sc, labels)
    db   = davies_bouldin_score(X_sc, labels)
    ch   = calinski_harabasz_score(X_sc, labels)

    st.subheader("🎯 K-Means Clustering Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score ↑", f"{sil:.4f}",
                help="Range -1 to 1. Higher = better separated clusters.")
    col2.metric("Davies-Bouldin ↓",   f"{db:.4f}",
                help="Lower = more compact & well-separated clusters.")
    col3.metric("Calinski-Harabasz ↑",f"{ch:.2f}",
                help="Higher = denser, well-separated clusters.")

    # Metric bars
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, val, name, color in zip(
        axes,
        [sil, db, ch/100],
        ['Silhouette ↑', 'Davies-Bouldin ↓', 'CH Score ↑ /100'],
        ['#2ecc71','#e74c3c','#3498db']):
        ax.bar([name], [val], color=color, edgecolor='white', width=0.4)
        ax.text(0, val + abs(val)*0.03, f'{val:.4f}',
                ha='center', fontsize=12, fontweight='bold')
        ax.set_title(name)
    plt.suptitle('K-Means Evaluation Metrics', fontsize=14)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔍 Cluster Size Distribution")
    cluster_counts = df_feat['KMeans_Cluster'].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar([f'C{k}:\n{CLUSTER_LABELS[k][0].split(" ",1)[1]}' for k in cluster_counts.index],
                cluster_counts.values,
                color=CLUSTER_COLORS, edgecolor='white')
    axes[0].set_title('Restaurants per Cluster')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(cluster_counts.values):
        axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

    axes[1].pie(cluster_counts.values,
                labels=[CLUSTER_LABELS[k][0] for k in cluster_counts.index],
                autopct='%1.1f%%', colors=CLUSTER_COLORS,
                startangle=140, wedgeprops=dict(edgecolor='white'))
    axes[1].set_title('Cluster Share')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📊 Cluster Profile Comparison")
    profile_cols = [c for c in ['Cost_INR_capped','Avg_Rating','Review_Count','Avg_Sentiment'] if c in df_feat.columns]
    profile = df_feat.groupby('KMeans_Cluster')[profile_cols].mean().round(2)
    profile.index = [CLUSTER_LABELS[k][0] for k in profile.index]
    st.dataframe(profile.style.background_gradient(cmap='RdYlGn', axis=0),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Feature Importance (Centroid Variance)")
    key_cols = [c for c in ['Cost_INR_capped','Avg_Rating','log_Review_Count',
                             'Avg_Sentiment','Pct_Positive','Avg_Pictures'] if c in df_feat.columns]
    centroid_df = pd.DataFrame(
        scaler.inverse_transform(df_feat_num.values[:4] if len(df_feat_num) >= 4 else df_feat_num.values),
        columns=feature_cols_avail)
    centroid_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols_avail)
    key_cols_avail = [c for c in key_cols if c in centroid_df.columns]
    cv = centroid_df[key_cols_avail].var(axis=0).sort_values(ascending=False)
    label_map = {'Cost_INR_capped':'Cost(INR)','Avg_Rating':'Rating',
                 'log_Review_Count':'Reviews(log)','Avg_Sentiment':'Sentiment',
                 'Pct_Positive':'%Positive','Avg_Pictures':'Pictures(log)'}
    cv.index = [label_map.get(c, c) for c in cv.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    cv.plot(kind='bar', ax=ax, color='#e23744', edgecolor='white', alpha=0.85)
    ax.set_title('Feature Importance – Centroid Variance')
    ax.set_ylabel('Variance (higher = more discriminative)')
    ax.tick_params(axis='x', rotation=30)
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2, p.get_height()+cv.max()*0.01,
                f'{p.get_height():.2f}', ha='center', fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Evaluation Metrics — Business Interpretation")
    interp = pd.DataFrame({
        'Metric': ['Silhouette Score','Davies-Bouldin','Calinski-Harabasz'],
        'Value':  [f"{sil:.4f}", f"{db:.4f}", f"{ch:.2f}"],
        'Range':  ['-1 to 1 (higher=better)','0 to ∞ (lower=better)','0 to ∞ (higher=better)'],
        'Business Meaning': [
            'Restaurants within a segment are truly similar → reliable targeted recommendations',
            'Segments are compact & non-overlapping → clean marketing segments',
            'Dense, well-separated clusters → consistent restaurant groupings for product features',
        ]
    })
    st.dataframe(interp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESTAURANT EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗂️ Restaurant Explorer":
    st.title("🗂️ Restaurant Explorer")
    if not models_loaded:
        st.error("Models not loaded."); st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    cluster_filter = col1.multiselect(
        "Filter by Segment",
        options=list(CLUSTER_LABELS.keys()),
        default=list(CLUSTER_LABELS.keys()),
        format_func=lambda x: CLUSTER_LABELS[x][0])
    cost_range = col2.slider("Cost Range (₹)", 0, 5000, (0, 5000), step=100)
    rating_min = col3.slider("Minimum Avg Rating", 1.0, 5.0, 1.0, step=0.1)

    # Merge cluster info with meta
    display_df = df_feat[['Name','KMeans_Cluster','Cost_INR_capped',
                           'Avg_Rating','Review_Count','Avg_Sentiment','Pct_Positive']].copy()
    display_df = display_df[
        (display_df['KMeans_Cluster'].isin(cluster_filter)) &
        (display_df['Cost_INR_capped'].between(cost_range[0], cost_range[1])) &
        (display_df['Avg_Rating'] >= rating_min)
    ].copy()

    display_df['Segment'] = display_df['KMeans_Cluster'].map(
        lambda k: CLUSTER_LABELS[k][0])
    display_df['Sentiment %'] = (display_df['Pct_Positive'] * 100).round(1)
    display_df = display_df.rename(columns={
        'Name':'Restaurant','Cost_INR_capped':'Cost(₹)',
        'Avg_Rating':'Rating','Review_Count':'Reviews',
        'Avg_Sentiment':'Sentiment Score'
    })

    st.markdown(f"**{len(display_df)} restaurants** match your filters")
    st.dataframe(
        display_df[['Restaurant','Segment','Cost(₹)','Rating','Reviews',
                    'Sentiment Score','Sentiment %']]
        .sort_values('Rating', ascending=False)
        .reset_index(drop=True)
        .style.background_gradient(subset=['Rating','Sentiment Score'], cmap='RdYlGn'),
        use_container_width=True, height=450)

    st.markdown("---")
    # Restaurant detail card
    st.subheader("🔎 Restaurant Detail")
    selected_rest = st.selectbox("Select a restaurant for details:",
                                  sorted(df_feat['Name'].unique()))
    r_feat = df_feat[df_feat['Name'] == selected_rest].iloc[0]
    r_reviews = df_reviews[df_reviews['Restaurant'] == selected_rest]

    col1, col2 = st.columns([1, 2])
    with col1:
        k = int(r_feat['KMeans_Cluster'])
        label, color = CLUSTER_LABELS[k]
        st.markdown(f"""
        <div class="cluster-card" style="background:{color};">
            {label}
        </div>
        """, unsafe_allow_html=True)
        st.metric("💰 Cost for Two", f"₹{r_feat.get('Cost_INR_capped', 'N/A'):.0f}")
        st.metric("⭐ Avg Rating",   f"{r_feat.get('Avg_Rating', 0):.2f}")
        st.metric("📝 Total Reviews",f"{int(r_feat.get('Review_Count', 0)):,}")
        st.metric("😊 Avg Sentiment",f"{r_feat.get('Avg_Sentiment', 0):.3f}")

    with col2:
        if not r_reviews.empty and 'Sentiment_Label' in r_reviews.columns:
            sl = r_reviews['Sentiment_Label'].value_counts()
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].pie(sl, labels=sl.index, autopct='%1.0f%%',
                        colors=['#2ecc71','#e74c3c','#f39c12'],
                        startangle=140, wedgeprops=dict(edgecolor='white'))
            axes[0].set_title('Sentiment Split')

            if 'Review_Month' in r_reviews.columns:
                m_avg = r_reviews.groupby('Review_Month')['Rating_Clean'].mean()
                axes[1].plot(m_avg.index, m_avg.values, 'o-',
                             color='#e23744', linewidth=2)
                axes[1].set_title('Rating Trend by Month')
                axes[1].set_xlabel('Month'); axes[1].set_ylabel('Avg Rating')
                axes[1].set_ylim(1, 5)

            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Show sample reviews
        if not r_reviews.empty:
            st.subheader("💬 Sample Reviews")
            sample_r = r_reviews[['Reviewer','Review','Rating_Clean',
                                   'Sentiment_Label']].head(5)
            st.dataframe(sample_r.reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "🍽️ Zomato Restaurant Intelligence Platform &nbsp;|&nbsp; "
    "Unsupervised ML + VADER NLP &nbsp;|&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True)