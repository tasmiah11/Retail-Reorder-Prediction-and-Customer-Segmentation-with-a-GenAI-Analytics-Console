# Retail-Reorder-Prediction-and-Customer-Segmentation-with-a-GenAI-Analytics-Console
## Overview

This project analyzes grocery retail transaction data to predict product reorders, segment customers by shopping behavior, and generate business level insights using Generative AI. It builds a full pipeline from raw transactional data through feature engineering, machine learning, customer segmentation, and natural language analytics.

The goal is to predict which products a customer will reorder and to understand different types of shoppers so promotions and recommendations can be targeted more effectively.

## Dataset

The notebook uses the following CSV files stored in a local `Data/` directory.

- aisles.csv  
- departments.csv  
- products.csv  
- orders.csv  
- order_products__prior.csv  
- order_products__train.csv  

These files are merged to create order line level records enriched with aisle and department information.

## Workflow implemented in the notebook

1. Load Python libraries for data processing, visualization, machine learning, and OpenAI.
2. Load all CSV datasets and merge products with aisles and departments.
3. Join enriched product data with order and order product tables.
4. Create basket level and user level features and run numerical EDA with plots.
5. Remove outliers using the IQR method with factor 3 on selected columns.
6. Build the modeling table `model_df` with engineered user and product features.
7. Split the data into training and test sets and apply preprocessing with scaling and one hot encoding.
8. Train multiple classifiers to predict the `reordered` label.
9. Evaluate models using accuracy, precision, recall, F1 score, and ROC AUC.
10. Compute Precision@K and Recall@K from predicted probabilities.
11. Perform customer segmentation using PCA and KMeans.
12. Profile segments and identify top departments per segment.
13. Run a promotion uplift simulation for a selected segment and department.
14. Add GenAI features for segment explanations and natural language analytics.

## Data cleaning

Outliers were removed using the IQR method (factor = 3).

| Stage | Row Count |
|------|-----------|
| Before outlier removal | 33,819,106 |
| After cleaning `add_to_cart_order` | 33,593,942 |

## Reorder prediction model performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------|----------|-----------|--------|---------|--------|
| Logistic Regression (L2) | 0.722 | 0.744 | 0.849 | 0.793 | 0.770 |
| Logistic Regression (L1) | 0.722 | 0.744 | 0.849 | 0.793 | 0.770 |
| Logistic Regression (Elastic Net) | 0.722 | 0.744 | 0.849 | 0.793 | 0.770 |
| Decision Tree | 0.733 | 0.755 | 0.851 | 0.800 | 0.785 |
| Random Forest | 0.716 | 0.718 | 0.903 | 0.800 | 0.770 |
| Logistic Regression with PCA (10 components) | — | — | — | — | 0.7687 |

## Ranking quality

Using predicted probabilities with K = 5:

| Metric | Value |
|--------|-------|
| Precision@5 | 0.707 |
| Recall@5 | 0.539 |

This means about 70 percent of the top five recommended products are actually reordered, and those five capture roughly 54 percent of all true reorders.

## Customer segmentation

Customers are embedded using PCA and clustered with KMeans. Each segment is profiled using basket size, order frequency, reorder rate, and top departments. These profiles are used to define meaningful shopper groups for targeting and promotions.

## Promotion uplift simulation

The notebook simulates a promotion by selecting a customer segment and department and estimating the uplift in expected orders if purchase probability increases.

## Generative AI features

Two GenAI features are implemented.

The first generates manager friendly summaries of customer segments based on computed statistics and top departments.

The second is a natural language analytics console. A user can ask a business question in plain English. The model returns pandas code that produces a `result` DataFrame, which is executed in the notebook.

