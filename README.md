# Restaurant Review NLP — Topic Modeling & Rating Prediction

Unsupervised NLP project using **Latent Dirichlet Allocation (LDA)** to discover latent topics in Yelp restaurant reviews, followed by **multinomial logistic regression** to predict customer star ratings from topic proportions.

## Overview

What drives a 5-star review vs. a 1-star review? This project goes beyond keyword counting to learn the underlying *experience dimensions* present in review text — then quantifies how strongly each dimension predicts ratings.

**Pipeline:**
1. Extract bigrams from review text
2. Fit LDA models with k = 5, 8, and 10 topics
3. Select best k based on topic interpretability
4. Regress star ratings on topic proportions (multinomial logistic)
5. Identify most positively and negatively rated topics

## Dataset

| File | Description |
|---|---|
| `we8therecounts.csv` | Bigram frequency matrix (reviews × bigrams) |
| `we8thereratings.csv` | Star ratings (1–5) for each review |

**Source:** Yelp restaurant review subset

## Methods

| Step | Method |
|---|---|
| Topic modeling | Latent Dirichlet Allocation (sklearn) — k ∈ {5, 8, 10} |
| Model selection | Qualitative interpretability + redundancy analysis |
| Regression | Multinomial logistic regression (one topic dropped as reference to avoid multicollinearity) |
| Visualization | Bar charts of top bigrams per topic; coefficient plots |

## Key Results

- **Optimal k = 8 topics** — best interpretability/separation tradeoff
- Topics span: food quality, service speed, ambiance, value, hygiene, drinks, staff friendliness, and wait times
- **Positive rating predictors:** food quality, friendly staff, good value
- **Negative rating predictors:** rude service, long waits, hygiene concerns, high prices
- Coefficient signs align with topic semantics, validating the LDA output

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-LDA-gray)
![pandas](https://img.shields.io/badge/pandas-gray)
![numpy](https://img.shields.io/badge/numpy-gray)
![matplotlib](https://img.shields.io/badge/matplotlib-gray)

## How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd HW5

pip install scikit-learn pandas numpy matplotlib

jupyter notebook 52610_HW5.ipynb
```

> **Note:** Place `we8therecounts.csv` and `we8thereratings.csv` in the same directory before running.

## File Structure

```
HW5-nlp-topic-modeling/
├── 52610_HW5.ipynb           # Main notebook
├── we8therecounts.csv        # Review bigram counts (not included)
├── we8thereratings.csv       # Star ratings (not included)
└── README.md
```

## Concepts Demonstrated

- Unsupervised text mining with LDA
- Bigram representation of text
- Multicollinearity handling in multinomial regression (reference category omission)
- Bridging unsupervised features → supervised prediction

---
*Course: MGMT 52610 — Data and AI-Driven Marketing*
