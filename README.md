# Instacart Market Basket Reorder Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This project is an end-to-end implementation of a machine learning system to solve the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) challenge. The goal is to predict which previously purchased products a user will reorder in their next shopping cart.

The solution progresses through the entire data science lifecycle, from exploratory data analysis and advanced feature engineering to model tuning and final deployment as a live, scalable web API on Google Cloud.

## 🚀 Project Highlights

- **High-Performance Model:** A tuned **XGBoost Classifier** that achieves a **F1 Score of 0.4506** on a held-out validation set.
- **Advanced Feature Engineering:** Created over 30 features capturing user behavior, product popularity, and high-resolution temporal patterns (e.g., "days since last purchase").
- **Dynamic Predictions:** Implemented a "Top-N" strategy to personalize the number of recommended items for each user based on their historical behavior.
- **Scalable Deployment:** The final model is deployed as a real-time prediction API using **Docker** and **Google Cloud Run**, with **Google BigQuery** serving as a scalable feature store.

## 🧰 Tech Stack

- **Language:** Python 3.9+
- **Core Libraries:** Pandas, Scikit-learn, XGBoost
- **Tuning:** Optuna
- **Deployment:** Flask, Gunicorn, Docker
- **Cloud:** Google Cloud Run, Google BigQuery

## 📁 Repository Structure

```
.
├── artifacts/      # Final, cleaned artifacts needed for the API to run
├── data/           # Raw data (to be downloaded from Kaggle)
├── deployment/     # Code and config for the live application (app.py, Dockerfile)
├── notebooks/      # Jupyter notebooks for EDA, feature engineering, and modeling
├── output/         # Archived experimental results and submission files
└── README.md
```

## 🛠️ How to Run This Project

### 1. Prerequisites

- Python 3.9 or higher
- Docker Desktop
- Access to a Google Cloud Platform project (for cloud deployment)

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Instacart-Market-Basket-Analysis.git
cd Instacart-Market-Basket-Analysis

# Install dependencies (ideally in a virtual environment)
pip install -r deployment/requirements.txt
```

You will also need to download the raw data from the [Kaggle competition page](https://www.kaggle.com/c/instacart-market-basket-analysis/data) and place the CSV files into the `data/` directory.

### 3. Running the Analysis

The development process is documented in the Jupyter notebooks located in the `notebooks/` directory. These notebooks cover everything from data exploration to final model training and artifact generation.

### 4. Running the API Locally with Docker

This is the best way to test the final application on your local machine before deploying it.

```bash
# From the project root directory, build the Docker image
docker build -t instacart-recommender -f deployment/Dockerfile .

# Run the container, mapping port 5000
docker run -d -p 5000:5000 instacart-recommender
```

Once the container is running, you can test the API by sending a POST request to `http://127.0.0.1:5000/predict`.

#### Using PowerShell

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -ContentType 'application/json' -Body '{"user_id": 1}'
```

#### Using cURL

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"user_id": 1}' http://127.0.0.1:5000/predict
```
