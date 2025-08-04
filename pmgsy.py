# Install required packages (run in IBM Cloud notebook cell)
# !pip install xgboost scikit-learn matplotlib seaborn sqlalchemy imbalanced-learn flask

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
from flask import Flask, request, jsonify
import joblib
import os
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# ==============================================
# DATABASE SETUP FOR SCHEME CLASSIFICATION
# ==============================================
def setup_database():
    """Create and populate SQLite database with scheme examples"""
    conn = sqlite3.connect('pmgsy_schemes.db')
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS scheme_examples")
    cursor.execute('''CREATE TABLE scheme_examples
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      road_length REAL,
                      cost REAL,
                      terrain TEXT,
                      population INTEGER,
                      scheme TEXT,
                      cost_per_km REAL,
                      pop_density REAL)''')
    
    examples = [
        (5.2, 1.8, 'Plain', 1200, 'PMGSY-I', 0.346, 230.77),
        (8.7, 3.2, 'Hilly', 800, 'PMGSY-II', 0.368, 91.95),
        (12.3, 4.5, 'Mountainous', 1500, 'RCPLWEA', 0.366, 121.95),
        (3.5, 1.2, 'Plain', 600, 'PMGSY-I', 0.343, 171.43),
        (15.0, 5.0, 'Mountainous', 2000, 'RCPLWEA', 0.333, 133.33)
    ]
    
    cursor.executemany('INSERT INTO scheme_examples VALUES (NULL,?,?,?,?,?,?,?)', examples)
    conn.commit()
    conn.close()

# ==============================================
# MODEL TRAINING FUNCTION
# ==============================================
def train_model():
    np.random.seed(42)
    n_samples = 3000

    def generate_terrain_data(terrain, n):
        if terrain == 'Plain':
            length = np.random.uniform(2, 15, n)
            cost = length * np.random.uniform(0.2, 0.4, n)
            pop = np.random.randint(800, 3000, n)
            scheme = np.random.choice(['PMGSY-I', 'PMGSY-II'], n, p=[0.85, 0.15])
        elif terrain == 'Hilly':
            length = np.random.uniform(3, 12, n)
            cost = length * np.random.uniform(0.3, 0.6, n)
            pop = np.random.randint(500, 2000, n)
            scheme = np.random.choice(['PMGSY-I', 'PMGSY-II'], n, p=[0.35, 0.65])
        else:  # Mountainous
            length = np.random.uniform(5, 20, n)
            cost = length * np.random.uniform(0.5, 0.8, n)
            pop = np.random.randint(300, 1500, n)
            scheme = np.random.choice(['PMGSY-II', 'RCPLWEA'], n, p=[0.25, 0.75])
        return length, cost, pop, scheme

    plain_len, plain_cost, plain_pop, plain_scheme = generate_terrain_data('Plain', n_samples//3)
    hilly_len, hilly_cost, hilly_pop, hilly_scheme = generate_terrain_data('Hilly', n_samples//3)
    mount_len, mount_cost, mount_pop, mount_scheme = generate_terrain_data('Mountainous', n_samples//3)

    df = pd.DataFrame({
        'road_length_km': np.concatenate([plain_len, hilly_len, mount_len]),
        'construction_cost_cr': np.concatenate([plain_cost, hilly_cost, mount_cost]),
        'terrain_type': ['Plain']*len(plain_len) + ['Hilly']*len(hilly_len) + ['Mountainous']*len(mount_len),
        'population_served': np.concatenate([plain_pop, hilly_pop, mount_pop]),
        'PMGSY_SCHEME': np.concatenate([plain_scheme, hilly_scheme, mount_scheme])
    })

    df['cost_per_km'] = df['construction_cost_cr'] / df['road_length_km']
    df['population_density'] = df['population_served'] / df['road_length_km']
    df = pd.get_dummies(df, columns=['terrain_type'], drop_first=True)

    label_encoder = LabelEncoder()
    df['scheme_encoded'] = label_encoder.fit_transform(df['PMGSY_SCHEME'])

    X = df.drop(['PMGSY_SCHEME', 'scheme_encoded'], axis=1)
    y = df['scheme_encoded']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    scaler = StandardScaler()
    num_cols = ['road_length_km', 'construction_cost_cr', 'population_served', 'cost_per_km', 'population_density']
    X_res[num_cols] = scaler.fit_transform(X_res[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    model = XGBClassifier(
        objective='multi:softmax',
        eval_metric=['merror', 'mlogloss'],
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        min_child_weight=2,
        early_stopping_rounds=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Save model and artifacts
    joblib.dump(model, 'pmgsy_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(X.columns, 'feature_columns.pkl')
    
    return model, scaler, label_encoder, X.columns

# ==============================================
# FLASK API ENDPOINTS
# ==============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Load artifacts
        model = joblib.load('pmgsy_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # Prepare input
        input_df = pd.DataFrame([{
            'road_length_km': data['road_length_km'],
            'construction_cost_cr': data['construction_cost_cr'],
            'terrain_type': data['terrain_type'],
            'population_served': data['population_served']
        }])
        
        # Feature engineering
        input_df['cost_per_km'] = input_df['construction_cost_cr'] / input_df['road_length_km']
        input_df['population_density'] = input_df['population_served'] / input_df['road_length_km']
        
        # One-hot encode
        input_df = pd.get_dummies(input_df, columns=['terrain_type'], drop_first=True)
        
        # Ensure all columns
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        
        # Reorder and scale
        input_df = input_df[feature_columns]
        num_cols = ['road_length_km', 'construction_cost_cr', 'population_served', 'cost_per_km', 'population_density']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Predict
        pred_encoded = model.predict(input_df)[0]
        pred = label_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(input_df)[0]
        
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
        
        return jsonify({
            'predicted_scheme': pred,
            'probabilities': prob_dict,
            'confidence': float(np.max(proba))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/examples', methods=['GET'])
def get_examples():
    conn = sqlite3.connect('pmgsy_schemes.db')
    query = '''
    SELECT road_length, cost, terrain, population, scheme, 
           ROUND(cost_per_km, 3) as cost_per_km,
           ROUND(pop_density, 2) as pop_density
    FROM scheme_examples
    ORDER BY terrain, scheme
    '''
    examples = pd.read_sql(query, conn).to_dict('records')
    conn.close()
    return jsonify(examples)

# ==============================================
# INITIALIZATION
# ==============================================
if __name__ == '__main__':
    # Setup database and train model if not already done
    if not os.path.exists('pmgsy_schemes.db'):
        setup_database()
    
    if not os.path.exists('pmgsy_model.pkl'):
        print("Training model...")
        train_model()
        print("Model trained and saved")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))