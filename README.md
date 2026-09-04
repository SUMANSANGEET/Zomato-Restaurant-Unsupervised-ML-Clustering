# 🍽️ Zomato Restaurant Intelligence Platform

### Unsupervised Machine Learning • Customer Sentiment • Restaurant Segmentation • Interactive Analytics

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-3776AB)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/)

</p>

<p align="center">

### 🚀 <a href="https://by72slunqsz2xkpbjakygq.streamlit.app/">Launch Live Interactive Dashboard</a>

  |  

### 💻 <a href="https://github.com/SUMANSANGEET/Zomato-Restaurant-Unsupervised-ML-Clustering">View Source Code</a>

</p>

---

## 📌 Executive Summary

The **Zomato Restaurant Intelligence Platform** is an end-to-end data science project that transforms restaurant metadata, customer reviews, ratings, pricing information, cuisines, and engagement signals into actionable business intelligence.

The project combines:

* 📊 Exploratory Data Analysis
* 🧹 Data Cleaning & Preprocessing
* ⚙️ Feature Engineering
* 📏 Feature Scaling
* 🤖 K-Means Unsupervised Learning
* 💬 NLP-based Sentiment Analysis
* 📈 Clustering Model Evaluation
* 🎯 Restaurant Segment Profiling
* 🚀 Interactive Streamlit Deployment

The objective is not simply to build a machine learning model, but to answer a practical business question:

> **How can restaurant data be segmented into meaningful customer-facing and business-facing groups to support pricing, marketing, positioning, and operational decisions?**

---

# 🎯 Business Problem

Restaurant platforms generate large volumes of structured and unstructured data.

Traditional descriptive analytics can tell us:

* Which restaurants have high ratings?
* Which cuisines are popular?
* What is the average cost?
* Which restaurants receive more reviews?

But businesses need deeper intelligence:

> **Which restaurants behave similarly, what differentiates their segments, and what strategic action should be taken for each segment?**

This project addresses that problem through **unsupervised machine learning and NLP-based sentiment intelligence**.

---

# 💡 Project Objectives

| Objective                       | Business Value                               |
| ------------------------------- | -------------------------------------------- |
| 🔍 Identify restaurant segments | Enables targeted business strategies         |
| 💰 Analyze pricing patterns     | Supports pricing and positioning decisions   |
| ⭐ Study restaurant ratings      | Measures perceived customer experience       |
| 🍴 Analyze cuisine preferences  | Identifies market demand                     |
| 💬 Analyze customer sentiment   | Goes beyond numerical ratings                |
| 🤖 Predict restaurant clusters  | Automates restaurant segmentation            |
| 📊 Build interactive analytics  | Makes insights accessible to decision-makers |

---

# 🧠 Solution Architecture

```text
                    ZOMATO RESTAURANT DATA
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Data Collection      │
                  │ Metadata + Reviews   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Cleaning        │
                  │ Missing Values       │
                  │ Duplicates           │
                  │ Data Types           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  │ Ratings              │
                  │ Cost                 │
                  │ Reviews              │
                  │ Cuisine Features     │
                  └──────────┬───────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌────────────────┐       ┌─────────────────┐
       │ EDA & Insights │       │ NLP Sentiment   │
       │                │       │ Analysis        │
       └───────┬────────┘       └────────┬────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Standard Scaling     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ K-Means Clustering   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Cluster Evaluation   │
                 │ Silhouette Score     │
                 │ Davies-Bouldin       │
                 │ Calinski-Harabasz    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Segment Profiling    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Dashboard  │
                 └──────────────────────┘
```

---

# 🤖 Machine Learning Approach

## K-Means Clustering

The project uses **K-Means clustering** to discover latent restaurant segments without predefined labels.

### Workflow

```text
Raw Restaurant Data
       ↓
Cleaning
       ↓
Feature Engineering
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
K-Means
       ↓
Cluster Assignment
       ↓
Cluster Profiling
       ↓
Business Interpretation
```

The trained model and preprocessing artifacts are persisted using **Joblib**, enabling the deployed application to perform predictions without retraining the model each time.

---

# 🧩 Restaurant Segmentation

The resulting restaurant groups are interpreted from their behavioral and commercial characteristics.

| Segment                    | Business Interpretation                     | Recommended Strategy                             |
| -------------------------- | ------------------------------------------- | ------------------------------------------------ |
| 💎 **Premium Fine Dining** | Higher-value positioning and strong ratings | Premium experiences, curated offers              |
| 🍕 **Popular Casual**      | Strong engagement and broad appeal          | Loyalty programs and retention                   |
| 💰 **Budget Gems**         | Affordable restaurants with potential       | Visibility and discovery campaigns               |
| ⚠️ **Underperformers**     | Weaker performance indicators               | Operational and customer-experience improvements |

> **Note:** Cluster numbers are model-generated labels; their business meaning comes from post-clustering profiling rather than the numeric cluster ID itself.

---

# 💬 NLP Sentiment Intelligence

Restaurant reviews provide qualitative information that numerical ratings cannot fully capture.

The application uses **VADER-based sentiment analysis** to classify review text into:

```text
😊 Positive
😐 Neutral
😞 Negative
```

### Why this matters

A restaurant can have an acceptable average rating while reviews reveal recurring complaints about:

* Service
* Food quality
* Waiting time
* Pricing
* Ambience
* Customer experience

This creates an additional layer of **customer-experience intelligence**.

---

# 📊 Interactive Analytics

The Streamlit application provides multiple analytical capabilities.

## 🏠 Business Dashboard

Provides a high-level view of:

* Restaurant KPIs
* Segment distribution
* Rating patterns
* Pricing characteristics
* Strategic observations

---

## 📈 Exploratory Data Analysis

Interactive visual analysis covering:

### Rating Intelligence

* Rating distribution
* Rating segmentation
* Rating vs. pricing relationships

### Pricing Intelligence

* Cost distribution
* Price segmentation
* Cost vs. rating relationships

### Cuisine Intelligence

* Popular cuisines
* Cuisine-level patterns
* Cuisine and pricing relationships

### Relationship Analysis

* Feature correlations
* Restaurant characteristics
* Engagement patterns

---

# 🎯 Cluster Prediction

Users can enter restaurant characteristics into the deployed application and obtain a predicted restaurant segment.

### Example workflow

```text
Restaurant Characteristics
          ↓
Feature Transformation
          ↓
Saved Scaler
          ↓
Saved K-Means Model
          ↓
Predicted Cluster
          ↓
Business Segment
```

This transforms the analytical model into a practical **decision-support tool**.

---

# 📈 Model Evaluation

Clustering quality is evaluated using multiple complementary metrics.

### Silhouette Score

Measures how well observations fit within their assigned clusters compared with neighboring clusters.

**Higher is generally better.**

### Davies-Bouldin Index

Measures similarity between clusters.

**Lower is generally better.**

### Calinski-Harabasz Score

Evaluates the ratio of between-cluster dispersion to within-cluster dispersion.

**Higher is generally better.**

Using multiple metrics provides a more robust assessment than relying on a single score.

---

# 🔍 Key Business Insights

## 💰 1. Pricing Strategy

The restaurant market contains strong competition across price segments.

**Business implication:** Restaurants should differentiate through value proposition rather than competing purely on price.

---

## ⭐ 2. Ratings Are Not the Complete Story

Numerical ratings summarize customer experience but can hide the specific reasons behind satisfaction or dissatisfaction.

**Business implication:** Combining ratings with review sentiment provides richer customer-experience intelligence.

---

## 🍴 3. Cuisine Concentration

A relatively small group of cuisine categories can account for substantial customer interest.

**Business implication:** Restaurants can use cuisine demand signals to identify competitive opportunities and underserved niches.

---

## 🎯 4. Segment-Specific Marketing

Different restaurant clusters require different strategies.

```text
Premium
   ↓
Experience + premium positioning

Popular Casual
   ↓
Retention + loyalty

Budget
   ↓
Visibility + discovery

Underperforming
   ↓
Operational improvement
```

---

# 🚀 Live Interactive Application

### Try the deployed application

👉 **[Launch Zomato Restaurant Intelligence Platform](https://by72slunqsz2xkpbjakygq.streamlit.app/)**

The live application enables users to interact with the project's:

* 📊 Analytics
* 🤖 Cluster prediction
* 💬 Sentiment analysis
* 🗂 Restaurant exploration
* 📈 Model evaluation

---

# 🖥️ Application Highlights

| Module                 | Capability                             |
| ---------------------- | -------------------------------------- |
| 🏠 Dashboard           | Business-level restaurant intelligence |
| 📊 EDA                 | Interactive exploratory analysis       |
| 🤖 Cluster Prediction  | Restaurant segment prediction          |
| 💬 Sentiment Analyzer  | Review sentiment classification        |
| 🗂 Restaurant Explorer | Restaurant filtering and investigation |
| 📈 Model Performance   | Clustering validation metrics          |

---

# 🛠️ Technology Stack

### Programming & Analysis

* Python
* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Machine Learning

* Scikit-learn
* K-Means Clustering
* StandardScaler

### NLP

* NLTK
* VADER Sentiment Analysis

### Model Persistence

* Joblib

### Application & Deployment

* Streamlit

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 📂 Repository Structure

```text
Zomato-Restaurant-Unsupervised-ML-Clustering/
│
├── 📓 Zomato Data Clustering Project.ipynb
│
├── 🚀 Zomato_app.py
│
├── 📋 requirements.txt
├── 📖 README.md
├── 🚫 .gitignore
│
├── 📊 Zomato Restaurant names and Metadata.csv
├── 💬 Zomato Restaurant reviews.csv
│
├── 📊 df_feat.csv
├── 📊 df_meta.csv
├── 📊 df_reviews.csv
│
├── 🤖 kmeans_final.joblib
├── 🤖 kmeans_model.joblib
├── 🤖 kmeans.joblib
├── ⚙️ scaler.joblib
├── ⚙️ feature_cols.joblib
├── 🍴 top10_cuisines.joblib
│
├── 🔧 refit_scaler.py
│
└── 📦 ZOMATO PROJECT.zip
```

The repository currently contains these project assets, including the notebook, Streamlit application, datasets, model artifacts, and dependency file.

---

# ⚙️ Installation & Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/SUMANSANGEET/Zomato-Restaurant-Unsupervised-ML-Clustering.git
```

## 2. Navigate to the project

```bash
cd Zomato-Restaurant-Unsupervised-ML-Clustering
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch Streamlit

```bash
streamlit run Zomato_app.py
```

The application will open in your browser.

---

# 📊 Data Science Skills Demonstrated

This project demonstrates practical capability across the complete analytics lifecycle:

```text
Business Problem Definition
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Statistical Analysis
        ↓
Unsupervised ML
        ↓
NLP
        ↓
Model Evaluation
        ↓
Business Interpretation
        ↓
Interactive Deployment
```

### Core competencies demonstrated

✅ Data preprocessing
✅ Exploratory data analysis
✅ Feature engineering
✅ Unsupervised machine learning
✅ K-Means clustering
✅ Model validation
✅ NLP sentiment analysis
✅ Data visualization
✅ Model persistence
✅ Streamlit application development
✅ Business insight generation
✅ End-to-end project deployment

---

# 💼 Recruiter Value

This project demonstrates more than notebook-based analysis.

It shows the ability to:

> **Transform raw restaurant data → engineer meaningful features → discover hidden customer/business segments → evaluate ML quality → extract business insights → deploy an interactive decision-support application.**

### Particularly relevant roles

* Data Analyst
* Data Scientist
* Business Analyst
* Machine Learning Intern
* Data Science Intern
* Product Analytics Intern
* Business Intelligence Analyst
* Junior ML Engineer

---

# 🔮 Future Enhancements

Potential next-generation improvements include:

* 🗺️ Interactive restaurant geospatial mapping
* 🎯 Personalized restaurant recommendation engine
* 🧠 Transformer-based sentiment analysis
* 📈 Restaurant trend forecasting
* 🔄 Automated model retraining pipeline
* ☁️ Cloud database integration
* 👥 Customer-level segmentation
* 🔎 Explainable cluster recommendations
* 📱 Mobile-optimized analytics experience

---

# 📌 Project Impact

The platform demonstrates how restaurant data can be transformed into an **actionable intelligence layer**.

Instead of asking only:

> "Which restaurant has the highest rating?"

the platform enables deeper questions:

> **"What type of restaurant is this?"**

> **"Which restaurants behave similarly?"**

> **"What does customer sentiment reveal?"**

> **"Which segment should receive a particular business strategy?"**

This makes the project relevant to real-world **restaurant analytics, marketplace intelligence, customer experience analytics, and strategic decision-making**.

---

# 👨‍💻 Author

## P. Suman Sangeet

**PGDM – Big Data Analytics | Data Science & Machine Learning**

Interested in building practical data products that combine:

**Data → Machine Learning → Visualization → Business Intelligence**

### 🔗 Project Links

* 🚀 **Live Streamlit App:** https://by72slunqsz2xkpbjakygq.streamlit.app/
* 💻 **GitHub Repository:** https://github.com/SUMANSANGEET/Zomato-Restaurant-Unsupervised-ML-Clustering

---

## ⭐ If You Find This Project Useful

Consider:

⭐ Starring the repository
🍴 Forking the project
💡 Sharing feedback
🤝 Connecting for collaboration

---

<p align="center">

### 🍽️ From Restaurant Data to Business Intelligence

**Analyze • Segment • Understand • Act**

</p>
