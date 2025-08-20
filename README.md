# Customer Churn Prediction Engine

> A smart, data-driven web application designed to forecast customer churn, empowering businesses to act before it's too late.

This project leverages a machine learning model to analyze customer data and predict the likelihood of a customer discontinuing their service. By providing a simple web interface, it makes complex predictive analytics accessible, allowing stakeholders to get instant insights and implement targeted retention strategies.

---

## 🚀 Screenshots

| Main Interface                                       |
| ---------------------------------------------------- | 
|<img width="957" height="693" alt="image" src="https://github.com/user-attachments/assets/c948f938-399c-4aa6-b214-d216fef5956e" />|

*(Note: Replace the image paths with actual screenshots of your application.)*

---

## ✨ Core Features

-   **Intuitive Web Interface:** A clean and straightforward form to input customer data for prediction.
-   **Instant Predictions:** Utilizes a pre-trained Scikit-learn model to deliver churn probability in real-time.
-   **Actionable Insights:** The output clearly states whether a customer is likely to "Churn" or "Stay," helping guide business decisions.
-   **RESTful API Backend:** Built with a lightweight Flask backend that serves the model and handles requests efficiently.
-   **Scalable Foundation:** The decoupled frontend and backend make it easy to extend features or integrate with other systems.

---

## 🛠️ Technology Stack

This project was built using a combination of data science and web development technologies:

-   **Backend:** **Python** with the **Flask** micro-framework.
-   **Machine Learning:**
    -   **Scikit-learn:** For building and training the predictive model.
    -   **Pandas:** For data manipulation and preprocessing.
    -   **NumPy:** For numerical operations.
-   **Frontend:** Standard **HTML**, **CSS**, and **JavaScript** for the user interface.
-   **Model Persistence:** The trained model is saved and loaded using Python's `pickle` module.

---

## ⚙️ Getting Started

To get a local copy up and running, please follow these steps.

### Prerequisites

Ensure you have Python 3.8 or newer installed on your machine.
-   [Download Python](https://www.python.org/downloads/)

### Installation & Setup

1.  **Clone the GitHub Repository:**
    ```sh
    git clone [https://github.com/your-username/churn-prediction-engine.git](https://github.com/your-username/churn-prediction-engine.git)
    cd churn-prediction-engine
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Required Libraries:**
    All necessary packages are listed in the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```

---

## 🚀 How to Use

1.  **Run the Flask Server:**
    Execute the main application file from your terminal.
    ```sh
    python app.py
    ```

2.  **Access the Application:**
    Open your favorite web browser and navigate to the following address:
    ```
    [http://127.0.0.1:5000](http://127.0.0.1:5000)
    ```

3.  **Make a Prediction:**
    -   Fill in the customer details in the web form.
    -   Click the **"Predict Churn"** button.
    -   The application will display the prediction result on the same page.

---

## 📁 Project Structure

The repository is organized as follows:

.├── app.py                  # The core Flask application logic├── model.pkl               # The pre-trained, serialized ML model├── requirements.txt        # A list of all Python dependencies├── templates/│   └── index.html          # The HTML file for the frontend UI└── static/└── css/└── style.css       # Stylesheet for the application
---
