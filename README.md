# Instacart Product Recommendation with Implicit ALS

A learning project that builds a basic grocery product recommender from past purchases. The notebook converts Instacart order histories into a user–product interaction matrix, trains an implicit-feedback Alternating Least Squares (ALS) model, and retrieves product recommendations for an example user.

Inspired by the [Building a MovieLens Recommender System workshop](https://www.youtube.com/watch?v=XfAe-HLysOM&t=3257s), I applied a more industry-oriented collaborative filtering approach to **Instacart** purchase histories: building a sparse user–product interaction matrix and training an **implicit ALS** model to generate recommendations.

## What the notebook implements

The [`RMcode1.ipynb`](notebooks/RMcode1.ipynb) notebook:

1. Loads Instacart's orders, prior and training order-product records, product catalog, departments, and aisles, then merges the tables for exploration.
2. Examines missing values and duplicate rows, fills missing `days_since_prior_order` in the prior-order data with zero, and filters to users with **more than five prior product rows**. The filter counts product rows in the notebook, not distinct orders.
3. Aggregates how often each retained user purchased each product and builds a sparse user–product matrix from those counts.
4. Fits `implicit.als.AlternatingLeastSquares` with **64 factors**, **0.1 regularization**, and **20 iterations**; asks the model for **10 product recommendations** for one example user and maps the results back to Instacart product IDs.

The saved notebook output shows a matrix of **204,741 users × 49,677 products**. This is a demonstration of training and generating recommendations; it does not include a held-out test set, ranking metrics, or a cold-start strategy.

## Reproduce the notebook

1. Download the six CSV files used by the notebook from the [Instacart Market Basket Analysis data](https://www.kaggle.com/competitions/basket-analysis/data): `orders.csv`, `products.csv`, `order_products__prior.csv`, `order_products__train.csv`, `departments.csv`, and `aisles.csv`.
2. Open [`notebooks/RMcode1.ipynb`](notebooks/RMcode1.ipynb) in Jupyter and change the six `C:\Users\songu\Downloads\archive (1)\...` paths in the first data-loading cell to the location of your downloaded CSV files.
3. Install the notebook's Python dependencies (`pandas`, `numpy`, `matplotlib`, `scipy`, `implicit`) and run its cells in order. The prior-order merge shown in the saved run has about **32.4 million rows**, so running it requires substantial memory.

**Data:** Instacart Market Basket Analysis. **Code:** Python, pandas, NumPy, SciPy sparse matrices, and `implicit` ALS. The source data is not stored in this repository.
