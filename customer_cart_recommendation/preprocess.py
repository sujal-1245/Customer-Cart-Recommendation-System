# preprocess.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os

# Create models folder if not exist
os.makedirs('models', exist_ok=True)

# 1. Load the dataset
file_path = 'customer_cart_data.csv'
data = pd.read_csv(file_path)

# 2. Preprocessing
data['Date'] = pd.to_datetime(data['Date'])  # Convert Date to datetime

# 3. Collaborative Filtering (KNN based)
print("[INFO] Building Collaborative Filtering Model...")
pivot_table = data.pivot_table(index='CustomerID', columns='Product', values='Quantity', fill_value=0)

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(pivot_table)

# Save pivot table and model
joblib.dump(pivot_table, 'models/pivot_table.pkl')
joblib.dump(knn, 'models/knn_model.pkl')

# 4. Association Rule Mining (Apriori)
print("[INFO] Building Association Rule Model...")
transaction_data = data.groupby(['CustomerID', 'Date', 'Product'])['Quantity'].sum().unstack().fillna(0)
transaction_data = transaction_data.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(transaction_data, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Save frequent itemsets and rules
frequent_itemsets.to_csv('models/frequent_itemsets.csv', index=False)
rules.to_csv('models/association_rules.csv', index=False)

print("[INFO] Preprocessing and Model Training Completed Successfully!")
