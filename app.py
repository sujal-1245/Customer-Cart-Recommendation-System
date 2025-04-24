from flask import Flask, jsonify, request, render_template
import pickle
from collections import Counter

app = Flask(__name__)

# Load model data
with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)

co_occurrence = data['co_occurrence']
customer_history = data['customer_history']
customer_sessions = data['customer_sessions']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(sorted(list(customer_history.keys())))

@app.route('/products/<int:customer_id>', methods=['GET'])
def get_customer_products(customer_id):
    return jsonify(sorted(list(set(customer_history.get(customer_id, [])))))

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    customer_id = content['customer_id']
    selected_product = content['selected_product']

    recommendations = Counter()
    recommendations.update(co_occurrence.get(selected_product, {}))

    sessions = customer_sessions.get(customer_id, [])
    for session in sessions:
        if selected_product in session:
            for prod in session:
                if prod != selected_product:
                    recommendations[prod] += 3

    top_n = [prod for prod, _ in recommendations.most_common(5)]
    return jsonify(top_n)

if __name__ == '__main__':
    app.run(debug=True)
