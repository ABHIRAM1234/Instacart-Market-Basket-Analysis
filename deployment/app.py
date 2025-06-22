# app.py (Final, Simplified Version)
from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import os
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- The Simplified Configuration ---
# All artifact files are expected to be in the same directory as this script.
MODEL_PATH = "instacart_xgb_model.json"
FEATURE_STORE_PATH = "feature_store.parquet"
PRODUCT_LOOKUP_PATH = "products_lookup.parquet"

# --- Load artifacts from the local directory ---
app.logger.info("Loading all production artifacts...")
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    products_lookup = pd.read_parquet(PRODUCT_LOOKUP_PATH)
    
    model_columns = model.get_booster().feature_names
    
    app.logger.info("Loading complete.")
except Exception as e:
    app.logger.error(f"FATAL: Could not load artifacts. {e}", exc_info=True)
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok" if model else "unhealthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded. Service is unhealthy."}), 503

    try:
        data = request.get_json()
        user_id = int(data.get('user_id'))
        app.logger.info(f"Received prediction request for user_id: {user_id}")

        user_features_df = feature_store[feature_store['user_id'] == user_id].copy()

        if user_features_df.empty:
            return jsonify({"user_id": user_id, "predicted_products": []})

        n = int(user_features_df['expected_reorders_n'].iloc[0])
        X_pred = user_features_df[model_columns]

        probabilities = model.predict_proba(X_pred)[:, 1]
        user_features_df['reorder_probability'] = probabilities
        
        if n == 0:
            return jsonify({"user_id": user_id, "predicted_products": []})

        top_n_products = user_features_df.sort_values(by='reorder_probability', ascending=False).head(n)
        
        results_df = top_n_products.merge(products_lookup, on='product_id', how='left')
        predictions = results_df[['product_id', 'product_name']].to_dict(orient='records')
        
        app.logger.info(f"Successfully predicted {len(predictions)} products for user {user_id}.")
        return jsonify({"user_id": user_id, "predicted_products": predictions})

    except Exception as e:
        app.logger.error(f"An error occurred during prediction for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)