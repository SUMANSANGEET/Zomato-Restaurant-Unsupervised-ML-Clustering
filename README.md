# Zomato-Restaurant-Unsupervised-ML-Clustering
# 🍽️ Zomato Restaurant Intelligence Platform

### *Turning Restaurant Data into Actionable Business Insights with Machine Learning and NLP*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/NLP-VADER-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

---

## 📖 Project Story

Restaurant platforms like **Zomato** generate massive volumes of data every day—ratings, reviews, cuisines, pricing patterns, customer engagement, and sentiment.

But an important question remains:

> **Can we uncover hidden restaurant segments and transform raw data into strategic business intelligence?**

This project answers that question by combining **Unsupervised Machine Learning** and **Natural Language Processing** to identify restaurant clusters, understand customer sentiment, and deliver interactive insights through a Streamlit dashboard.

---

## 🎯 Objectives

This project aims to:

* 🔍 Discover hidden restaurant segments using clustering techniques.
* 📊 Analyze customer behavior through ratings and reviews.
* 😊 Understand customer emotions using sentiment analysis.
* 🍴 Explore cuisine preferences and pricing trends.
* 🚀 Build an interactive platform for restaurant intelligence.

---

## 🌟 Key Features

### 📊 Interactive Analytics Dashboard

Explore restaurant trends through dynamic visualizations.

### 🤖 Restaurant Cluster Prediction

Predict the business segment of a restaurant based on its characteristics.

### 💬 Real-Time Sentiment Analyzer

Analyze restaurant reviews instantly using **VADER NLP**.

### 🗂 Restaurant Explorer

Filter and investigate restaurants by ratings, pricing, sentiment, and clusters.

### 📈 Model Evaluation Dashboard

Understand clustering performance using multiple evaluation metrics.

---

# 🔍 Business Questions Addressed

✔ What restaurant segments exist within the market?

✔ Which restaurants drive the highest engagement?

✔ How does sentiment align with ratings?

✔ Which cuisines dominate consumer preferences?

✔ Which restaurants require strategic intervention?

---

# ⚙️ Machine Learning Workflow

```text
Restaurant Data
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Standard Scaling
       │
       ▼
K-Means Clustering
       │
       ▼
Cluster Interpretation
       │
       ▼
Business Recommendations
       │
       ▼
Interactive Streamlit Dashboard
```

---

# 🧠 Restaurant Segments Identified

| Cluster      | Segment             | Insights                                     |
| ------------ | ------------------- | -------------------------------------------- |
| 💎 Cluster 0 | Premium Fine Dining | High ratings, higher spending customers      |
| 🍕 Cluster 1 | Popular Casual      | Strong engagement and mass appeal            |
| 💰 Cluster 2 | Budget Gems         | Affordable restaurants with growth potential |
| ⚠️ Cluster 3 | Underperformers     | Restaurants needing strategic improvements   |

---

# 📊 Dashboard Preview

The Streamlit application includes:

### 🏠 Home Dashboard

* Business KPIs
* Segment summaries
* Strategic insights

### 📊 EDA Dashboard

* Cost distribution analysis
* Cuisine popularity trends
* Rating distributions
* Correlation analysis

### 🤖 Cluster Prediction

Input restaurant characteristics and predict their segment instantly.

### 💬 Sentiment Analysis

Paste customer reviews and receive:

```text
😊 Positive
😐 Neutral
😞 Negative
```

along with sentiment scores.

### 📈 Model Performance

Evaluate clustering quality using:

* Silhouette Score
* Davies–Bouldin Index
* Calinski–Harabasz Score

### 🗂 Restaurant Explorer

Discover restaurants through powerful filters.

---

# 🛠 Technologies Used

| Category          | Tools              |
| ----------------- | ------------------ |
| Programming       | Python             |
| Data Analysis     | Pandas, NumPy      |
| Visualization     | Matplotlib, Plotly |
| Machine Learning  | Scikit-learn       |
| NLP               | NLTK, VADER        |
| Deployment        | Streamlit          |
| Model Persistence | Joblib             |

---

# 📂 Repository Structure

```text
📦 Zomato-Restaurant-Intelligence
│
├── app.py
├── requirements.txt
├── scaler.joblib
├── kmeans_final.joblib
├── feature_cols.joblib
├── top10_cuisines.joblib
├── df_feat.csv
├── df_meta.csv
├── df_reviews.csv
├── notebooks/
│   └── Zomato_Analysis.ipynb
│
├── images/
│
└── README.md
```

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/yourusername/zomato-restaurant-intelligence.git
```

## Navigate to the Project Folder

```bash
cd zomato-restaurant-intelligence
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch the Streamlit App

```bash
streamlit run app.py
```

---

# 🌐 Live Application

### 🔗 Streamlit Demo

```text
https://your-app-name.streamlit.app
```

*(Update this link after deployment.)*

---

# 💡 Business Insights Generated

## 💰 Pricing Strategy

Most restaurants operate within the **mid-price segment**, indicating strong competition and the importance of differentiation.

---

## 😊 Sentiment Intelligence

Customer ratings alone may not fully reflect customer satisfaction.

Sentiment analysis uncovers hidden dissatisfaction even among moderate ratings.

---

## 🍴 Cuisine Trends

A small number of cuisines dominate the market, while premium cuisines command significantly higher prices.

---

## 🎯 Targeted Marketing

Different restaurant segments require different business strategies:

* Premium experiences for fine dining.
* Loyalty programs for popular casual restaurants.
* Visibility campaigns for budget segments.
* Operational improvements for underperformers.

---

# 📈 Model Evaluation

The clustering solution was assessed using multiple validation techniques to ensure meaningful segmentation.

Metrics used include:

```text
• Silhouette Score
• Davies–Bouldin Index
• Calinski–Harabasz Score
```

These metrics help determine how distinct and cohesive the identified restaurant groups are.

---

# 🔮 Future Enhancements

* [ ] Recommendation System
* [ ] Deep Learning-based Sentiment Analysis
* [ ] Interactive Geographic Mapping
* [ ] Automated Retraining Pipeline
* [ ] Cloud-based Database Integration
* [ ] Restaurant Trend Forecasting

---

# 🤝 Contributions

Contributions are welcome.

If you have suggestions for improving this project:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

---

# ⭐ If You Found This Project Interesting...

Consider supporting it by:

```text
⭐ Starring the repository
🍴 Forking the project
📢 Sharing it with others
```

Your support helps the project reach more learners and practitioners interested in applying Machine Learning to real-world business problems.

---

# 👨‍💻 Author

### **P. Suman Sangeet**

**Aspiring Data Scientist | Machine Learning Enthusiast | Building practical AI solutions through data-driven insights**

---

> *"Data becomes valuable when it guides decisions. This project transforms restaurant data into intelligence that businesses can act upon."*
