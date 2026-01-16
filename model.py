import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class BreastCancerModel:
    """
    Breast Cancer Prediction Model
    Predicts whether a tumor is Malignant (M) or Benign (B)
    """

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        # Top 10 most important features for prediction
        self.feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
        ]

    def load_data(self, file_path='data/data.csv'):
        """
        Load and preprocess breast cancer data
        Returns: X (features), y (diagnosis labels)
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} patient records")

            # Drop id column (not needed for prediction)
            if 'id' in df.columns:
                df = df.drop('id', axis=1)

            # Check for missing values
            missing = df.isnull().sum().sum()
            if missing > 0:
                print(f"  Warning: {missing} missing values found, filling with median")
                df = df.fillna(df.median())

            # Encode diagnosis: M (Malignant) = 1, B (Benign) = 0
            df['diagnosis_encoded'] = self.label_encoder.fit_transform(df['diagnosis'])

            # Use top 10 features for simplicity (you can use all 30 if you want)
            X = df[self.feature_names]
            y = df['diagnosis_encoded']

            print("\nDataset Statistics:")
            print(f"  Total Patients: {len(df)}")
            malignant_count = (df['diagnosis'] == 'M').sum()
            benign_count = (df['diagnosis'] == 'B').sum()
            print(f"  Malignant (Cancerous): {malignant_count} ({malignant_count/len(df)*100:.1f}%)")
            print(f"  Benign (Non-cancerous): {benign_count} ({benign_count/len(df)*100:.1f}%)")

            return X, y

        except FileNotFoundError:
            print(f"✗ Error: {file_path} not found")
            print("  Make sure 'data.csv' is in the 'data/' folder")
            return None, None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None, None

    def train(self, X, y):
        """
        Train the breast cancer prediction model
        Uses Random Forest Classifier
        """
        print("\n--- Training Model ---")

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        print("\n  Training in progress...")
        self.model.fit(X_train, y_train)
        print("  ✓ Training completed!")

        # Evaluate model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Testing Accuracy: {test_accuracy * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_predictions)
        print(f"\nConfusion Matrix:")
        print(f"  True Benign: {cm[0][0]}")
        print(f"  False Malignant (False Positive): {cm[0][1]}")
        print(f"  False Benign (False Negative): {cm[1][0]}")
        print(f"  True Malignant: {cm[1][1]}")

        # Detailed classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_predictions,
                                    target_names=['Benign', 'Malignant']))

        # Feature importance
        self._show_feature_importance()

        return test_accuracy

    def _show_feature_importance(self):
        """Display which features matter most for cancer prediction"""
        if self.model:
            importance = self.model.feature_importances_
            print("\nFeature Importance (What matters most for prediction):")
            feature_imp = sorted(zip(self.feature_names, importance),
                                 key=lambda x: x[1], reverse=True)
            for name, imp in feature_imp:
                bar = "█" * int(imp * 100)
                print(f"  {name:25s}: {bar} {imp:.4f}")

    def predict(self, radius_mean, texture_mean, perimeter_mean, area_mean,
                smoothness_mean, compactness_mean, concavity_mean,
                concave_points_mean, symmetry_mean, fractal_dimension_mean):
        """
        Predict cancer diagnosis for a patient

        Parameters (Top 10 features):
        - radius_mean: Mean of distances from center to points on the perimeter
        - texture_mean: Standard deviation of gray-scale values
        - perimeter_mean: Mean size of the core tumor
        - area_mean: Mean area of the tumor
        - smoothness_mean: Mean of local variation in radius lengths
        - compactness_mean: Mean of perimeter^2 / area - 1.0
        - concavity_mean: Mean of severity of concave portions
        - concave_points_mean: Mean number of concave portions
        - symmetry_mean: Mean symmetry
        - fractal_dimension_mean: Mean "coastline approximation" - 1

        Returns: (prediction, probability, diagnosis)
        - prediction: 0 (Benign) or 1 (Malignant)
        - probability: Probability of being Malignant (0-1)
        - diagnosis: 'B' (Benign) or 'M' (Malignant)
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Run train() first.")

        # Create feature array
        features = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean
        ]])

        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        # Convert back to diagnosis label
        diagnosis = self.label_encoder.inverse_transform([prediction])[0]

        return prediction, probability[1], diagnosis  # Return malignant probability

    def save_model(self, model_dir='model_files'):
        """Save trained model and encoder to disk"""
        if self.model is None:
            print("✗ No model to save. Train the model first.")
            return False

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.label_encoder, encoder_path)

            print(f"\n✓ Model saved to {model_path}")
            print(f"✓ Label encoder saved to {encoder_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
            return False

    def load_model(self, model_dir='model_files'):
        """Load pre-trained model and encoder from disk"""
        model_path = os.path.join(model_dir, 'breast_cancer_model.pkl')
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"✓ Model loaded from {model_path}")
            print(f"✓ Label encoder loaded successfully")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found in {model_dir}")
            print("  Run train_and_save_model() first to create the model")
            return False
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False


def train_and_save_model():
    """
    Main training function - run this to create the model
    """
    print("=" * 70)
    print("BREAST CANCER PREDICTION MODEL TRAINING")
    print("=" * 70)

    # Initialize model
    cancer_model = BreastCancerModel()

    # Load data
    X, y = cancer_model.load_data()
    if X is None:
        print("\n✗ Training aborted due to data loading error")
        return None

    # Train model
    accuracy = cancer_model.train(X, y)

    # Save trained model
    cancer_model.save_model()

    # Test predictions with various scenarios
    print("\n" + "=" * 70)
    print("TESTING SAMPLE PREDICTIONS")
    print("=" * 70)

    test_cases = [
        {
            'name': 'Typical Benign Case (Low values)',
            'radius_mean': 12.0, 'texture_mean': 15.0, 'perimeter_mean': 80.0,
            'area_mean': 450.0, 'smoothness_mean': 0.09, 'compactness_mean': 0.08,
            'concavity_mean': 0.05, 'concave_points_mean': 0.03,
            'symmetry_mean': 0.17, 'fractal_dimension_mean': 0.06
        },
        {
            'name': 'Typical Malignant Case (High values)',
            'radius_mean': 20.0, 'texture_mean': 25.0, 'perimeter_mean': 130.0,
            'area_mean': 1200.0, 'smoothness_mean': 0.12, 'compactness_mean': 0.25,
            'concavity_mean': 0.30, 'concave_points_mean': 0.15,
            'symmetry_mean': 0.25, 'fractal_dimension_mean': 0.08
        },
        {
            'name': 'Borderline Case (Mixed values)',
            'radius_mean': 15.0, 'texture_mean': 20.0, 'perimeter_mean': 95.0,
            'area_mean': 700.0, 'smoothness_mean': 0.10, 'compactness_mean': 0.15,
            'concavity_mean': 0.12, 'concave_points_mean': 0.07,
            'symmetry_mean': 0.20, 'fractal_dimension_mean': 0.07
        }
    ]

    for i, case in enumerate(test_cases, 1):
        pred, prob, diagnosis = cancer_model.predict(
            radius_mean=case['radius_mean'],
            texture_mean=case['texture_mean'],
            perimeter_mean=case['perimeter_mean'],
            area_mean=case['area_mean'],
            smoothness_mean=case['smoothness_mean'],
            compactness_mean=case['compactness_mean'],
            concavity_mean=case['concavity_mean'],
            concave_points_mean=case['concave_points_mean'],
            symmetry_mean=case['symmetry_mean'],
            fractal_dimension_mean=case['fractal_dimension_mean']
        )
        
        result = "Malignant (Cancerous) ⚠️" if diagnosis == 'M' else "Benign (Non-cancerous) ✓"
        print(f"\n{i}. {case['name']}")
        print(f"   Prediction: {result}")
        print(f"   Malignancy Probability: {prob * 100:.1f}%")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Final Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 70)
    print("\nYou can now use the trained model in your Flask app!")
    
    return cancer_model


if __name__ == "__main__":
    # Train and save the model
    model = train_and_save_model()