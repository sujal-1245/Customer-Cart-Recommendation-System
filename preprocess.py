import pandas as pd
import pickle
from collections import defaultdict, Counter
from itertools import combinations

# Load data
df = pd.read_csv('dataset_group.csv', names=['date', 'customer_id', 'product'], header=None, skiprows=1)
df['date'] = pd.to_datetime(df['date'])

# Group by customer and date for sessions
sessions = df.groupby(['customer_id', 'date'])['product'].apply(list)
transactions = sessions.tolist()

# Co-occurrence matrix
co_occurrence = defaultdict(Counter)
for products in transactions:
    unique_products = set(products)
    for prod1, prod2 in combinations(unique_products, 2):
        co_occurrence[prod1][prod2] += 1
        co_occurrence[prod2][prod1] += 1

# Customer history
customer_history = df.groupby('customer_id')['product'].apply(list).to_dict()
customer_sessions = df.groupby('customer_id').apply(lambda x: x.groupby('date')['product'].apply(list).tolist()).to_dict()

# Save all to a pickle file
with open('model_data.pkl', 'wb') as f:
    pickle.dump({
        'co_occurrence': co_occurrence,
        'customer_history': customer_history,
        'customer_sessions': customer_sessions
    }, f)

print("✅ Preprocessing done. Model data saved.")
