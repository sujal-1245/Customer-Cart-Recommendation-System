# Customer Cart Recommendation System

## Overview
The **Customer Cart Recommendation System** provides personalized product recommendations using collaborative filtering and association rule mining. It suggests items based on the products in a customer's cart and suggests product bundles, enhancing the shopping experience and increasing potential sales.

## 📷 Screenshot
![image](https://github.com/user-attachments/assets/c3cbcb2e-3ad6-42ef-a733-2a4457fac3cd)


## Features
- **Collaborative Filtering**: Uses K-Nearest Neighbors (KNN) to recommend products based on customer similarity.
- **Association Rule Mining**: Implements the Apriori algorithm to suggest bundles of products frequently bought together.
- **Flask-based Web Application**: Provides a user-friendly interface for customers and an API for recommendations and bundles.

## Technologies
- Python
- Flask
- Scikit-learn
- Pandas
- mlxtend (for Apriori)
- Joblib (for saving models)

## Project Structure
- **preprocess.py**: Loads and preprocesses the data, trains models for collaborative filtering and association rule mining, and saves them.
- **app.py**: Loads the models and provides functions to generate product recommendations and bundles.
- **recommender.py**: Contains the recommendation logic for KNN-based product suggestions and association rule-based bundle suggestions.
- **Flask Web App**: Provides an interface for users to get personalized recommendations and bundles.

## Setup Instructions
1. **Clone the repository**:
    ```bash
    git clone https://github.com/sujal-1245/Customer-Cart-Recommendation-System.git
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Preprocessing**:
    Ensure the dataset `customer_cart_data.csv` is in the same directory as `preprocess.py` and run:
    ```bash
    python preprocess.py
    ```

4. **Run the Flask app**:
    ```bash
    python app.py
    ```
    Visit `http://127.0.0.1:5000/` to access the web interface.

## Endpoints
- **GET `/api/recommend/<customer_id>`**: Get product recommendations for a customer.
- **GET `/api/bundle/<product_name>`**: Get product bundle suggestions for a given product.

