# Banknote Fraud Detection — Naive Bayes & PCA Classifier

A machine learning project that classifies scanned banknotes as **genuine** or 
**forged**, built to demonstrate applied data science, statistical reasoning, and 
business-aligned model selection for a fraud-risk use case.

---

## 📌 Business Problem

Retail banks process large volumes of physical cash through ATMs, branch 
counters, and cash-processing centres. Each note is scanned and a set of 
image-based features is generated to assess authenticity. Relying on manual 
visual inspection at this volume is slow and inconsistent, so an automated 
classifier is needed to flag suspected counterfeit notes for removal or further 
review before they re-enter circulation.

This project frames banknote authentication as a **binary classification 
problem**: using four measurable features from a banknote scan, predict whether 
a note is **genuine** or **forged**.

The cost of errors is asymmetric:

- **False Negative** (forged note classified genuine) → the note passes the 
  automated check and re-enters circulation, creating direct financial loss and 
  regulatory/audit exposure.
- **False Positive** (genuine note classified forged) → the note is routed to 
  manual review, creating operational cost and delay, but no monetary loss.

A missed forgery is more costly than an unnecessary manual review. The goal is 
therefore not simply maximum accuracy, but **minimising missed forgeries while 
keeping the manual review rate operationally manageable**.

---

## 🎯 Project Objectives

- Build a reliable classifier to distinguish genuine from forged banknotes using 
  scan-derived features.
- Ensure model evaluation is unbiased by separating training, validation, and 
  test data, and only cleaning the training set.
- Investigate whether dimensionality reduction (PCA) can simplify the feature 
  space without losing predictive signal.
- Determine the minimum number of raw features needed to achieve strong, 
  generalisable performance.
- Evaluate model performance through a business-risk lens, not just raw accuracy.

---

## 🔍 Data Overview & Preprocessing

The dataset contains 1,372 banknote scans, each described by four image-derived 
features and an authenticity label (Genuine / Forged). The data was profiled for 
missing values, duplicates, and class imbalance before any model was built. 
No missing values were found; 24 duplicate scans were identified, and the class 
split (55.5% Genuine / 44.5% Forged) was confirmed to be fairly balanced, 
requiring no resampling technique.

The dataset was split into training (60%), validation (20%), and test (20%) sets 
**before** any cleaning was applied, ensuring validation and test data remain an 
unbiased reflection of real, unfiltered scans. Duplicates and outliers were 
removed from the training set only — never from validation or test — so reported 
performance reflects how the model would behave on live data in production.

---

## 🧮 Dimensionality Reduction (PCA)

Principal Component Analysis was used to test whether the authenticity signal 
could be compressed into fewer, uncorrelated dimensions. The analysis showed that 
the first two components capture **85.55%** of total variance across the four 
original features — meaning most of what distinguishes a genuine note from a 
forged one can be summarised in just two underlying dimensions. The test set was 
projected onto this same space (without influencing how it was defined), and 
visual comparison confirmed the class separation seen in training data held 
consistently in unseen test data.

---

## 🤖 Modelling Approach

A Gaussian Naive Bayes classifier was trained on the PCA-reduced feature space 
and achieved a strong **96.36%** test accuracy, with **zero forged notes 
misclassified as genuine** — every forgery in the test set was caught, at the 
cost of a small number of genuine notes being flagged for manual review.

A second investigation tested how model performance changes as raw features are 
added one at a time, to determine whether a simpler, non-PCA model could perform 
comparably. This identified `variance` and `skewness` as the two most informative 
features, with additional features adding no meaningful improvement. A final 
model trained on just these two features achieved **88.36%** test accuracy.

---

## ⚖️ Model Comparison: Accuracy vs. Risk

Two valid models emerged, each optimising for a different business priority:

| | **PCA Model (4 components)** | **Raw Feature Model (2 features)** |
|---|---|---|
| Test Accuracy | 96.36% | 88.36% |
| Forged notes missed | **0** | 13 |
| Genuine notes flagged for review | 10 | 19 |
| Interpretability | Lower | Higher |

**Business takeaway:** the PCA model has a zero false-negative rate — no forged 
note slips through — making it the stronger choice where minimising fraud loss is 
the priority. The raw-feature model is simpler and easier to explain to 
non-technical stakeholders, trading some forgery-detection accuracy for 
operational simplicity. Which model to deploy depends on whether the 
organisation prioritises minimising fraud losses or interpretability and cost of 
deployment.

---

## ✅ Conclusion

The dataset was clean and reasonably balanced, with data-cleaning steps 
deliberately confined to the training set to preserve unbiased evaluation. PCA 
confirmed that most of the authenticity signal in this dataset can be captured in 
two dimensions, and a Naive Bayes model trained on the full PCA space achieved 
strong, fraud-safe performance with no missed forgeries. A simpler model using 
only two raw features achieved solid but lower accuracy, illustrating a genuine 
trade-off between predictive strength and interpretability. Naive Bayes proved to 
be an effective and explainable approach to banknote authentication, and the 
comparison between the two models highlights a recurring theme in applied data 
science: the "best" model depends on what the business is optimising for.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib`

---

## 📁 Repository Structure

```
bank-fraud-detection/
├── README.md
├── SOFTWARE_ARTEFACT.md       ← full technical write-up with code, outputs, and corrections
├── notebook/
│   └── banknote_authentication.ipynb
├── data/
│   └── bank_notes.csv
```

For the full technical implementation — code, outputs, and detailed explanations 
behind every step — see [`SOFTWARE_ARTEFACT.md`](./SOFTWARE_ARTEFACT.md).
