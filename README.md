# Customer Churn Prediction Application

A web-based application that leverages a machine learning model to predict customer churn. This tool provides businesses with actionable insights by identifying customers who are at high risk of leaving, allowing for proactive retention strategies.

<img width="957" height="693" alt="image" src="https://github.com/user-attachments/assets/c0fb074f-bac6-41f8-84ce-3de5e6a527ec" />

---

## ✨ Key Features

- **Predictive Analysis:** Utilizes a trained machine learning model to forecast churn probability based on customer data.
- **User-Friendly Interface:** A clean and simple UI for inputting customer details and viewing the prediction result.
- **Real-time Predictions:** Get instant churn predictions to facilitate quick decision-making.
- **Scalable Architecture:** Built with a modern tech stack that is easy to maintain and extend.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS
- **Backend:** Python with Flask 
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Data Handling:** The model was trained on a CSV dataset containing customer information and churn status.

---

## ⚙️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following installed on your system:
- Python 3.8+
- pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/churn-prediction-app.git](https://github.com/your-username/churn-prediction-app.git)
    cd churn-prediction-app
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

1.  **Run the Flask application:**
    ```sh
    python app.py
    ```

2.  **Open your web browser** and navigate to `http://127.0.0.1:5000`.

3.  **Enter the customer data** into the input form on the web page.

4.  **Click the "Predict" button** to see the churn prediction result. The application will display whether the customer is likely to churn or not.

---

## 📁 Project Structure

churn-prediction-app/
├── app.py                # Main Flask application file
├── model.pkl             # Pre-trained machine learning model
├── templates/
│   └── index.html        # HTML template for the user interface
├── static/
│   └── css/
│       └── style.css     # CSS for styling the application
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
---
