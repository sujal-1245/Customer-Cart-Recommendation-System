from flask import Flask, render_template, request, jsonify
from utils.recommender import recommend_products_knn, suggest_bundle, get_all_customers, get_all_products

app = Flask(__name__)

# Serve the frontend
@app.route('/')
def home():
    customers = get_all_customers()
    products = get_all_products()
    return render_template('index.html', customers=customers, products=products)

# API for frontend form (POST)
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    customer_id = int(data.get('customer_id'))
    recommendations = recommend_products_knn(customer_id)
    return jsonify({'recommendations': recommendations})

@app.route('/bundle', methods=['POST'])
def bundle():
    data = request.get_json()
    product_name = data.get('product_name')
    suggestions = suggest_bundle(product_name)
    return jsonify({'suggestions': suggestions})

# Optional old API (for testing if needed)
@app.route('/api/recommend/<int:customer_id>', methods=['GET'])
def api_recommend(customer_id):
    recommendations = recommend_products_knn(customer_id)
    return jsonify({
        "customer_id": customer_id,
        "recommended_products": recommendations
    })

@app.route('/api/bundle/<string:product_name>', methods=['GET'])
def api_bundle(product_name):
    bundle_suggestions = suggest_bundle(product_name)
    return jsonify({
        "product_name": product_name,
        "bundle_suggestions": bundle_suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)
