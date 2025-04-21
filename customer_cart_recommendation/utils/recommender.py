import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Helper to clean frozenset strings
def clean_frozenset_string(x):
    x = x.replace('frozenset(', '').replace(')', '')
    try:
        return set(eval(x))
    except:
        return set()

# Load saved models
pivot_table = joblib.load('models/pivot_table.pkl')
knn_model = joblib.load('models/knn_model.pkl')
association_rules = pd.read_csv('models/association_rules.csv')

# Clean antecedents and consequents
association_rules['antecedents'] = association_rules['antecedents'].apply(clean_frozenset_string)
association_rules['consequents'] = association_rules['consequents'].apply(clean_frozenset_string)

def recommend_products_knn(customer_id, n_recommendations=5):
    if customer_id not in pivot_table.index:
        return ["Customer not found."]
    
    customer_vector = pivot_table.loc[[customer_id]]
    distances, indices = knn_model.kneighbors(customer_vector, n_neighbors=n_recommendations + 1)
    
    recommendations = []
    for idx in indices.flatten()[1:]:  # Skip self (first neighbor)
        similar_customer_id = pivot_table.index[idx]
        recommended_products = pivot_table.loc[similar_customer_id]
        recommended_products = recommended_products[recommended_products > 0].index.tolist()
        recommendations.extend(recommended_products)

    already_bought = pivot_table.loc[customer_id]
    already_bought = already_bought[already_bought > 0].index.tolist()
    final_recommendations = list(set(recommendations) - set(already_bought))
    
    return final_recommendations[:n_recommendations]

def suggest_bundle(product_name):
    matching_rules = association_rules[association_rules['antecedents'].apply(lambda x: product_name in x)]
    
    if matching_rules.empty:
        return ["No bundle suggestion found."]
    
    suggestions = []
    for consequent in matching_rules['consequents']:
        suggestions.extend(list(consequent))
    
    suggestions = list(set(suggestions) - {product_name})
    
    return suggestions[:5]

def get_all_customers():
    return pivot_table.index.tolist()

def get_all_products():
    all_products = set()
    for items in association_rules['antecedents']:
        all_products.update(items)
    return sorted(all_products)
