# 🧠 Website Credibility Classification (Thesis)

## 🧩 The Problem I Chose
The internet is full of online media sources, but their credibility varies widely.  
My goal was to build a system that predicts **website credibility (high vs non-high)** based purely on **HTML structure and linguistic content**.

---

## ⚙️ What I Did
- 🕸️ **Extracted website data** from over 4,000 HTML pages.  
- 🧹 **Parsed content using BeautifulSoup** to collect titles, headings, meta tags, links, and text.  
- 🗂️ **Merged with media bias metadata** (factual reporting, bias rating, country, etc.).  
- 🧠 **Engineered features** from HTML and linguistic data using:
  - Tag structure frequency
  - Readability and sentiment
  - Empath semantic categories
- ⚖️ **Balanced classes using SMOTE** to address dataset imbalance.  
- 🧮 **Trained multiple classifiers** (Random Forest, XGBoost, SVM, Logistic Regression) for both 3-class and binary tasks.  
- 🧪 **Evaluated models** with accuracy, recall, F1, and calibration metrics.  
- 📈 **Visualized feature importance and confusion matrices** for interpretability.

---

## 📊 The Outcome
- Reformulated the task to **binary classification (high vs non-high credibility)** for better performance.  
- **Calibrated Logistic Regression model** achieved:
  - 87% accuracy  
  - Perfect recall on high-credibility websites  
  - Brier score improved from 0.21 → 0.1086  
- **Feature analysis** revealed key credibility indicators like:
  - Number of list tags  
  - Reading time  
  - Empath semantic tone categories  

---

## 🧠 Key Insights
- Websites with structured layouts and moderate text lengths tend to correlate with higher credibility.  
- HTML and linguistic cues can be powerful credibility predictors, even without user interaction data.  

---

## 🛠 Tech Stack
`Python` · `scikit-learn` · `BeautifulSoup` · `Empath` · `SMOTE` · `Matplotlib`  

---

## 📂 Repository Contents
