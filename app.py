from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and encoder
model = None
label_encoder = None

def load_model():
    """Load the pre-trained model and encoder"""
    global model, label_encoder
    
    model_dir = 'model_files'
    
    try:
        model = joblib.load(os.path.join(model_dir, 'breast_cancer_model.pkl'))
        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
        print("✓ Model loaded successfully!")
        print(f"✓ Model type: {type(model).__name__}")
        return True
    except FileNotFoundError:
        print("✗ Model files not found!")
        print("Please run 'python model.py' first to train and save the model")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the frontend HTML from templates folder"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract top 10 features
        radius_mean = float(data.get('radius_mean', 0))
        texture_mean = float(data.get('texture_mean', 0))
        perimeter_mean = float(data.get('perimeter_mean', 0))
        area_mean = float(data.get('area_mean', 0))
        smoothness_mean = float(data.get('smoothness_mean', 0))
        compactness_mean = float(data.get('compactness_mean', 0))
        concavity_mean = float(data.get('concavity_mean', 0))
        concave_points_mean = float(data.get('concave_points_mean', 0))
        symmetry_mean = float(data.get('symmetry_mean', 0))
        fractal_dimension_mean = float(data.get('fractal_dimension_mean', 0))
        
        # Prepare input for prediction
        input_features = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean
        ]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0]
        
        # Convert to diagnosis
        diagnosis = label_encoder.inverse_transform([prediction])[0]
        
        # Return result
        return jsonify({
            'diagnosis': diagnosis,
            'is_malignant': bool(prediction),
            'malignant_probability': float(probability[1]),
            'benign_probability': float(probability[0]),
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load the pre-trained model
    print("=" * 60)
    print("STARTING BREAST CANCER PREDICTION API")
    print("=" * 60)
    
    if load_model():
        print("\n✓ Server ready!")
        print("=" * 60)
        # Get port from environment variable (for Render) or use 5000 as default
        port = int(os.environ.get('PORT', 5000))
        # Run the Flask app
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("\n✗ Failed to start server")
        print("Please run 'python model.py' first to create the model files")
        print("=" * 60)