#  Website Credibility Classification (Thesis)

##  The Problem I Chose
The internet is full of online media sources, but their credibility varies widely.  
Distinguishing reliable outlets from misleading ones is essential for journalism, academia, and automated fact-checking.
My goal was to design a machine-learning system that predicts website credibility (high vs non-high) using only the **HTML structure** and **linguistic content** of each site — without relying on human labels or social metadata.
---

##  What I Did
I built an end-to-end data pipeline that transformed raw web data into structured, machine-readable insights.
-  **Extracted website data** from over 4,000 HTML pages each mapped to a Media Bias / Fact Check (MBFC) credibility label.  
-  **Parsed content using BeautifulSoup** to collect titles, headings, meta tags, links, and text.  
-  **Merged with media bias metadata** (factual reporting, bias rating, country, etc.).  
-  **Engineered features** from HTML and linguistic data using:
  - Tag structure frequency
  - Readability and sentiment
  - Sentiment polarity (from meta descriptions)
  - Empath semantic categories (~200+ topics such as trust, violence, law, politics)
-  **Balanced classes using SMOTE** to address dataset imbalance.  
-  **Trained multiple classifiers** (Random Forest, XGBoost, SVM, Logistic Regression) for both 3-class and binary tasks.  
-  **Evaluated models** with accuracy, recall, F1, and calibration metrics.  
-  **Visualized feature importance and confusion matrices** for interpretability.

---

##  The Outcome
- Reformulating the problem into binary classification (high vs non-high credibility) led to more consistent and interpretable results.
- Models using HTML + Empath + textual features achieved strong and balanced performance across all metrics.
- Tree-based models (Random Forest and XGBoost) performed slightly better overall, capturing complex non-linear patterns between HTML structure and linguistic cues.
- Logistic Regression remained highly interpretable, showing how individual features such as tag frequency, sentiment, and reading time contribute to credibility prediction.
- Model calibration further improved probability reliability, ensuring the system could provide trustworthy confidence scores.
- Overall, results confirmed that credibility can be predicted effectively from website structure and language alone, without relying on external user data or traffic metrics.

---

##  Key Insights
- HTML structure and linguistic tone can reliably reflect credibility without relying on social engagement metrics.
- Balanced data (via SMOTE) and probability calibration are critical for fair and trustworthy classification.
- Websites with clear structure, consistent metadata, and moderate text complexity are statistically more credible.
- The integration of Empath semantic categories provided valuable context about writing tone and content focus.

---

## 🛠 Tech Stack
`Python` · `scikit-learn` · `BeautifulSoup` · `Empath` · `SMOTE` · `Matplotlib`  

---

## 📂 Repository Contents
