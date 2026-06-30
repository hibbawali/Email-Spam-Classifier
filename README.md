# Email Spam Classifier

A machine learning project that detects spam messages using TF-IDF vectorization and compares two classification models — Logistic Regression and Naive Bayes — achieving 97% accuracy.

## Project Overview
- Dataset: SMS Spam Collection (Kaggle) — 5,572 messages
- Models compared: Naive Bayes vs Logistic Regression
- Best model: Naive Bayes (97.1% accuracy, 0.99 precision, 0.79 recall)
- Built a web interface where users can paste any message and get instant spam/ham prediction

## Key Findings
- Dataset was imbalanced (87% ham, 13% spam) — accuracy alone was misleading
- Error analysis revealed model struggled with adult-content spam due to vocabulary mismatch
- Naive Bayes outperformed Logistic Regression on recall (0.79 vs 0.72)

## Tech Stack
- Python, Scikit-learn, Pandas, Flask
- TF-IDF Vectorization
- Joblib (model saving)

## Live Demo 
- Website Link (https://spamshield-web.vercel.app/)

## Dataset
[SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Internship
Built during Machine Learning internship at Arch Technologies.
