🚀 PMGSY Scheme Classification using XGBoost & SMOTE
📌 Project Overview
This project implements an Intelligent Classification System for rural infrastructure projects under the Pradhan Mantri Gram Sadak Yojana (PMGSY).
The system uses XGBoost, SMOTE, and Flask to classify projects into their respective schemes:

PMGSY-I

PMGSY-II

RCPLWEA

The solution is designed to help policymakers and planners with efficient fund allocation, transparent monitoring, and impact assessment.

📂 Repository Structure
bash
Copy
Edit
├── pmgsy.py          # Main application script with ML model & Flask API  
├── IBM.pdf           # Project report including problem statement, system approach, and results  
├── requirements.txt  # Required Python libraries  
├── README.md         # Project documentation (this file)  
🛠️ Features
Data Simulation & Preprocessing

Generates synthetic project data for different terrains (Plain, Hilly, Mountainous).

Feature engineering: cost per km & population density.

Balances dataset using SMOTE.

Machine Learning Model

XGBoost Classifier optimized with hyperparameters.

Accuracy ~88–90% on test data.

Provides prediction probabilities & feature importance visualization.

Flask REST API

/predict → Classify new project input (JSON).

/examples → View sample classified projects from SQLite database.

Visualization

Confusion Matrix for performance analysis.

Bar charts for prediction confidence.

Feature importance graph.

⚙️ Installation
1. Clone Repository
bash
Copy
Edit
git clone https://github.com/Krisha2005-qcode/PMGSY_IBM_35.git
cd PMGSY_IBM_35
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If running in IBM Watson Studio, add in a notebook cell:

python
Copy
Edit
!pip install xgboost scikit-learn matplotlib seaborn sqlalchemy imbalanced-learn flask joblib
▶️ Usage
Run Flask App
bash
Copy
Edit
python pmgsy.py
The server will start on:

cpp
Copy
Edit
http://127.0.0.1:5000
Example Request
Send a POST request to /predict:

json
Copy
Edit
{
  "road_length_km": 8.5,
  "construction_cost_cr": 3.5,
  "terrain_type": "Hilly",
  "population_served": 1200
}
Example Response
json
Copy
Edit
{
  "predicted_scheme": "PMGSY-II",
  "probabilities": {
    "PMGSY-I": 0.12,
    "PMGSY-II": 0.78,
    "RCPLWEA": 0.10
  },
  "confidence": 0.78
}
📊 Results
Accuracy: ~90%

Strong performance across all three schemes.

Feature Importance: Cost per km, Population Density, Road Length are most influential.

📘 Documentation
Full project documentation is available in IBM.pdf.
