# To run this application:
# 1. Install necessary libraries: pip install Flask scikit-learn pandas joblib
# 2. Save this code as a single file (e.g., app.py).
# 3. Run the file from your terminal: python app.py
# 4. Open your web browser and go to http://127.0.0.1:5000

import pandas as pd
import joblib
from flask import Flask, render_template_string, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# --- Part 1: Model Training and Saving (Run this part once) ---
# This section simulates the training process and saves the necessary files.
# In a real project, this would be a separate training script.

def train_and_save_model():
    """
    Trains a Random Forest model on the Telco Churn dataset and saves
    the model, scaler, and model columns to disk.
    """
    # Load the dataset from a reliable URL
    df = pd.read_csv("D:/project/Telco-Customer-Churn.csv")

    # --- Data Cleaning ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    # --- Preprocessing ---
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-Hot Encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Save the column order for later use in prediction
    joblib.dump(X_encoded.columns, 'model_columns.joblib')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')

    # --- Model Training (using the best parameters we found) ---
    # We use class_weight to handle the imbalance in the dataset
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0: 1, 1: 5}, # Penalize misclassifying churners more
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save the trained model
    joblib.dump(model, 'churn_model.joblib')

    print("Model, scaler, and columns have been trained and saved.")

# --- Part 2: Flask Application ---

# Initialize the Flask app
app = Flask(__name__)

# --- HTML Template ---
# A modern UI with Tailwind CSS for styling the input form.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="container mx-auto max-w-3xl p-8 bg-white rounded-2xl shadow-xl">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800">Customer Churn Prediction</h1>
            <p class="text-gray-500 mt-2">Enter customer details to predict the likelihood of churn.</p>
        </header>

        <!-- Prediction Result Display -->
        {% if prediction_text %}
        <div id="result" class="text-center p-4 mb-6 rounded-lg 
            {% if 'Not Likely' in prediction_text %} bg-green-100 text-green-800 
            {% else %} bg-red-100 text-red-800 {% endif %}">
            <h2 class="text-2xl font-semibold">{{ prediction_text }}</h2>
        </div>
        {% endif %}

        <!-- Input Form -->
        <form action="/predict" method="POST">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Numerical Inputs -->
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="text-lg font-semibold mb-3 text-gray-700 border-b pb-2">Account Information</h3>
                    <div>
                        <label for="tenure" class="block text-sm font-medium text-gray-600">Tenure (Months)</label>
                        <input type="number" id="tenure" name="tenure" required value="1" min="0" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    </div>
                    <div class="mt-4">
                        <label for="MonthlyCharges" class="block text-sm font-medium text-gray-600">Monthly Charges ($)</label>
                        <input type="number" step="0.01" id="MonthlyCharges" name="MonthlyCharges" required value="29.99" min="0" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    </div>
                    <div class="mt-4">
                        <label for="TotalCharges" class="block text-sm font-medium text-gray-600">Total Charges ($)</label>
                        <input type="number" step="0.01" id="TotalCharges" name="TotalCharges" required value="100.00" min="0" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                    </div>
                </div>

                <!-- Categorical Inputs -->
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="text-lg font-semibold mb-3 text-gray-700 border-b pb-2">Services & Contract</h3>
                    <div>
                        <label for="Contract" class="block text-sm font-medium text-gray-600">Contract Type</label>
                        <select id="Contract" name="Contract" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option>Month-to-month</option>
                            <option>One year</option>
                            <option>Two year</option>
                        </select>
                    </div>
                    <div class="mt-4">
                        <label for="InternetService" class="block text-sm font-medium text-gray-600">Internet Service</label>
                        <select id="InternetService" name="InternetService" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option>DSL</option>
                            <option>Fiber optic</option>
                            <option>No</option>
                        </select>
                    </div>
                    <div class="mt-4">
                        <label for="PaymentMethod" class="block text-sm font-medium text-gray-600">Payment Method</label>
                        <select id="PaymentMethod" name="PaymentMethod" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option>Electronic check</option>
                            <option>Mailed check</option>
                            <option>Bank transfer (automatic)</option>
                            <option>Credit card (automatic)</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Submit Button -->
            <div class="mt-8 text-center">
                <button type="submit" class="w-full md:w-1/2 py-3 px-6 border border-transparent rounded-md shadow-sm text-lg font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">
                    Predict Churn
                </button>
            </div>
        </form>
    </div>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives customer data from the form, preprocesses it, and returns a prediction.
    """
    # Load the model, scaler, and column names
    model = joblib.load('churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')

    # Get form data
    form_data = request.form.to_dict()
    
    # Convert form data to a DataFrame
    input_df = pd.DataFrame([form_data])

    # Convert numerical columns to the correct type
    input_df['tenure'] = pd.to_numeric(input_df['tenure'])
    input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'])
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])

    # --- Preprocessing the input data ---
    # One-Hot Encode the new data
    input_encoded = pd.get_dummies(input_df)

    # Align columns: ensure the input has the exact same columns as the training data
    # 'reindex' will add any missing columns and fill them with 0
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the aligned data using the loaded scaler
    input_scaled = scaler.transform(input_aligned)

    # --- Make Prediction ---
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- Format Output ---
    if prediction[0] == 1:
        churn_probability = prediction_proba[0][1] * 100
        result_text = f"Prediction: Likely to Churn ({churn_probability:.2f}% probability)"
    else:
        result_text = "Prediction: Not Likely to Churn"
    
    # Render the page again, but this time with the prediction result
    return render_template_string(HTML_TEMPLATE, prediction_text=result_text)


# --- Main Entry Point ---
if __name__ == '__main__':
    # Check if the model files exist. If not, train and create them.
    if not all(os.path.exists(f) for f in ['churn_model.joblib', 'scaler.joblib', 'model_columns.joblib']):
        print("Model files not found. Training a new model...")
        train_and_save_model()
    else:
        print("Found existing model files.")

    # Run the Flask web server
    print("Starting Flask server... Go to http://127.0.0.1:5000")
    app.run(debug=True)
