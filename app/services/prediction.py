"""XGBoost Success Probability Prediction Service"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuccessPredictionModel:
    """XGBoost-based success probability prediction model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.get("XGBOOST_MODEL_PATH", "models/xgboost_model.pkl")
        self.model = None
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
        # Create models directory
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _create_features(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for success prediction"""
        
        logger.info("Creating features for success prediction...")
        
        # Load original data for additional features
        from app.services.similarity import load_data
        students_df, internships_df = load_data()
        
        # Merge with student and internship data
        features_df = sim_df.copy()
        
        # Merge student features
        student_features = students_df[['id', 'gpa', 'education', 'category', 'past_internships', 'location']]
        features_df = features_df.merge(student_features, left_on='student_id', right_on='id', how='left', suffixes=('', '_student'))
        
        # Merge internship features  
        internship_features = internships_df[['id', 'company', 'capacity', 'location', 'reserved_for']]
        features_df = features_df.merge(internship_features, left_on='internship_id', right_on='id', how='left', suffixes=('', '_internship'))
        
        # Basic similarity features
        features_df['similarity_score_squared'] = features_df['similarity_score'] ** 2
        features_df['similarity_score_log'] = np.log1p(features_df['similarity_score'])
        
        # Student academic features
        features_df['gpa_normalized'] = (features_df['gpa'] - features_df['gpa'].min()) / (features_df['gpa'].max() - features_df['gpa'].min())
        features_df['high_achiever'] = (features_df['gpa'] >= 8.5).astype(int)
        features_df['academic_tier'] = pd.cut(features_df['gpa'], bins=[0, 6.5, 7.5, 8.5, 10], labels=[0, 1, 2, 3]).astype(int)
        
        # Experience features
        features_df['has_experience'] = (features_df['past_internships'] > 0).astype(int)
        features_df['experienced'] = (features_df['past_internships'] >= 2).astype(int)
        features_df['experience_tier'] = np.minimum(features_df['past_internships'], 3)  # Cap at 3+
        
        # Education level features
        features_df['is_masters'] = features_df['education'].str.contains('M\\.Tech|Master|MCA|MBA', case=False, na=False).astype(int)
        features_df['is_tech_degree'] = features_df['education'].str.contains('Tech|Engineering|Computer|IT|CSE', case=False, na=False).astype(int)
        
        # Category and diversity features
        features_df['is_rural'] = (features_df['category'] == 'rural').astype(int)
        features_df['is_reserved'] = (features_df['category'] == 'reserved').astype(int)
        features_df['is_female'] = (features_df['category'] == 'female').astype(int)
        features_df['is_general'] = (features_df['category'] == 'general').astype(int)
        
        # Internship features
        features_df['high_capacity'] = (features_df['capacity'] >= 5).astype(int)
        features_df['capacity_normalized'] = (features_df['capacity'] - features_df['capacity'].min()) / (features_df['capacity'].max() - features_df['capacity'].min())
        
        # Location matching
        features_df['location_match'] = (features_df['location'] == features_df['location_internship']).astype(int)
        
        # Company tier (based on capacity as proxy for company size)
        features_df['company_tier'] = pd.cut(features_df['capacity'], bins=[0, 3, 6, 100], labels=[0, 1, 2]).astype(int)
        
        # Skill matching features (basic text analysis)
        def skill_overlap_ratio(student_skills, required_skills):
            if pd.isna(student_skills) or pd.isna(required_skills):
                return 0.0
            
            student_set = set(str(student_skills).lower().split(','))
            required_set = set(str(required_skills).lower().split(','))
            
            if len(required_set) == 0:
                return 0.0
            
            overlap = len(student_set.intersection(required_set))
            return overlap / len(required_set)
        
        features_df['skill_overlap_ratio'] = features_df.apply(
            lambda row: skill_overlap_ratio(row.get('student_skills', ''), row.get('required_skills', '')), 
            axis=1
        )
        
        # Interaction features
        features_df['gpa_similarity_interaction'] = features_df['gpa_normalized'] * features_df['similarity_score']
        features_df['experience_similarity_interaction'] = features_df['experience_tier'] * features_df['similarity_score']
        features_df['education_similarity_interaction'] = features_df['is_masters'] * features_df['similarity_score']
        
        # Competition features (number of students competing for same internship)
        competition_counts = features_df.groupby('internship_id').size().reset_index(name='competition_level')
        features_df = features_df.merge(competition_counts, on='internship_id', how='left')
        features_df['competition_normalized'] = (features_df['competition_level'] - features_df['competition_level'].min()) / (features_df['competition_level'].max() - features_df['competition_level'].min())
        
        return features_df
    
    def _generate_synthetic_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic success labels based on realistic assumptions"""
        
        logger.info("Generating synthetic success labels...")
        
        # Base success probability calculation
        base_prob = 0.6  # Base success rate
        
        # Factors that increase success probability
        prob_adjustments = pd.Series(base_prob, index=features_df.index)
        
        # Academic performance impact
        prob_adjustments += (features_df['gpa_normalized'] - 0.5) * 0.3
        
        # Similarity score impact (most important factor)
        prob_adjustments += (features_df['similarity_score'] - 0.5) * 0.4
        
        # Experience impact
        prob_adjustments += features_df['experience_tier'] * 0.1
        
        # Education level impact
        prob_adjustments += features_df['is_masters'] * 0.1
        
        # Skill overlap impact
        prob_adjustments += features_df['skill_overlap_ratio'] * 0.2
        
        # Location matching bonus
        prob_adjustments += features_df['location_match'] * 0.05
        
        # Competition penalty (more competition = lower individual success chance)
        prob_adjustments -= features_df['competition_normalized'] * 0.1
        
        # Interaction effects
        prob_adjustments += features_df['gpa_similarity_interaction'] * 0.1
        prob_adjustments += features_df['experience_similarity_interaction'] * 0.05
        
        # Ensure probabilities are in [0, 1] range
        success_probabilities = np.clip(prob_adjustments, 0.1, 0.9)
        
        # Generate binary labels based on probabilities
        np.random.seed(42)  # For reproducibility
        success_labels = np.random.binomial(1, success_probabilities)
        
        features_df['success_probability'] = success_probabilities
        features_df['success_label'] = success_labels
        
        logger.info(f"Generated labels with success rate: {success_labels.mean():.3f}")
        
        return features_df
    
    def _prepare_features(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        
        # Select numeric features for training
        numeric_features = [
            'similarity_score', 'similarity_score_squared', 'similarity_score_log',
            'gpa_normalized', 'high_achiever', 'academic_tier',
            'has_experience', 'experienced', 'experience_tier',
            'is_masters', 'is_tech_degree',
            'is_rural', 'is_reserved', 'is_female', 'is_general',
            'high_capacity', 'capacity_normalized', 'company_tier',
            'location_match', 'skill_overlap_ratio',
            'gpa_similarity_interaction', 'experience_similarity_interaction', 
            'education_similarity_interaction', 'competition_normalized'
        ]
        
        # Filter available features
        available_features = [f for f in numeric_features if f in features_df.columns]
        self.feature_names = available_features
        
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        X = features_df[available_features].fillna(0)
        y = features_df['success_label'] if 'success_label' in features_df.columns else None
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, sim_df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model for success prediction"""
        
        logger.info("Starting XGBoost model training...")
        
        # Create comprehensive features
        features_df = self._create_features(sim_df)
        
        # Generate synthetic labels (in real scenario, these would be historical data)
        features_df = self._generate_synthetic_labels(features_df)
        
        # Prepare features
        X, y = self._prepare_features(features_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # XGBoost parameters from config
        xgb_params = settings.get("XGBOOST_PARAMS", {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })
        
        # Train model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(**xgb_params, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        logger.info(feature_importance.head(10).to_string(index=False))
        
        # Save model and components
        model_payload = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        self._save_model(model_payload)
        
        return model_payload
    
    def _save_model(self, model_payload: Dict[str, Any]):
        """Save trained model and components"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_payload, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> Dict[str, Any]:
        """Load trained model and components"""
        try:
            with open(self.model_path, 'rb') as f:
                model_payload = pickle.load(f)
            
            self.model = model_payload['model']
            self.feature_scaler = model_payload['feature_scaler']
            self.feature_names = model_payload['feature_names']
            
            logger.info(f"Model loaded from {self.model_path}")
            return model_payload
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def predict_success_probability(self, sim_df: pd.DataFrame, model_payload: Dict[str, Any] = None) -> pd.DataFrame:
        """Predict success probabilities for student-internship pairs"""
        
        if model_payload:
            self.model = model_payload['model']
            self.feature_scaler = model_payload['feature_scaler']
            self.feature_names = model_payload['feature_names']
        
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        logger.info("Predicting success probabilities...")
        
        # Create features (without generating labels)
        features_df = self._create_features(sim_df)
        
        # Prepare features for prediction
        available_features = [f for f in self.feature_names if f in features_df.columns]
        
        if len(available_features) != len(self.feature_names):
            logger.warning(f"Some features missing. Using {len(available_features)}/{len(self.feature_names)} features")
        
        X = features_df[available_features].fillna(0)
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict probabilities
        success_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        result_df = sim_df.copy()
        result_df['success_prob_pred'] = success_probabilities
        result_df['success_prob_pred'] = result_df['success_prob_pred'].round(4)
        
        logger.info(f"Success probability predictions completed. Mean probability: {success_probabilities.mean():.3f}")
        
        return result_df

# Global model instance
prediction_model = SuccessPredictionModel()

def train_xgb(sim_df: pd.DataFrame) -> Dict[str, Any]:
    """Train XGBoost model (backward compatibility)"""
    return prediction_model.train_model(sim_df)

def load_model() -> Dict[str, Any]:
    """Load XGBoost model (backward compatibility)"""
    return prediction_model.load_model()

def predict_success_prob(sim_df: pd.DataFrame, model_payload: Dict[str, Any]) -> pd.DataFrame:
    """Predict success probabilities (backward compatibility)"""
    return prediction_model.predict_success_probability(sim_df, model_payload)

if __name__ == "__main__":
    # Test the prediction model
    from app.services.similarity import run_similarity
    
    # Get similarity data
    _, sim_df = run_similarity(force_reload=False, top_k=20)
    
    # Train model
    model_payload = train_xgb(sim_df.head(1000))  # Use subset for testing
    
    # Test predictions
    predictions_df = predict_success_prob(sim_df.head(100), model_payload)
    
    print("✅ Success prediction model training and testing complete!")
    print(f"Predictions shape: {predictions_df.shape}")
    print(f"Mean success probability: {predictions_df['success_prob_pred'].mean():.3f}")
    print("\nSample predictions:")
    print(predictions_df[['student_id', 'internship_id', 'similarity_score', 'success_prob_pred']].head())
