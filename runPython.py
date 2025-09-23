"""
Complete Internship Allocation Pipeline
========================================

This script orchestrates the entire ML pipeline for internship allocation:
1. Data Loading and Preprocessing (Scikit-learn)
2. Similarity Computation (SentenceTransformers)  
3. Success Prediction (XGBoost)
4. Optimization (OR-Tools)
5. Validation and Analytics

Features:
- Handles large-scale data (100K+ students, 10K+ internships)
- Advanced caching for performance
- Comprehensive analytics and reporting
- Fairness constraints and diversity quotas
- Multiple optimization strategies with fallbacks

Usage:
    python runPython.py --help
    python runPython.py --train-model --force-embeddings
    python runPython.py --top-k-sim 50 --top-k-opt 30 --quota 0.20
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.similarity import run_similarity, load_data, similarity_engine
from app.services.prediction import train_xgb, load_model, predict_success_prob, prediction_model  
from app.services.optimizer import run_optimizer
from app.api.analytics import generate_allocation_report
from app.core.config import settings

import logging

# Enhanced logging setup
def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging"""
    
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class AllocationPipeline:
    """Complete allocation pipeline with advanced features"""
    
    def __init__(self):
        self.execution_stats = {
            'start_time': None,
            'stages': {},
            'total_time': None,
            'success': False
        }
        
        # Create output directories
        for directory in ["data", "models", "analytics", "logs"]:
            os.makedirs(directory, exist_ok=True)
    
    def _log_stage_start(self, stage_name: str):
        """Log the start of a pipeline stage"""
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {stage_name}")
        logger.info(f"{'='*60}")
        self.execution_stats['stages'][stage_name] = {'start_time': time.time()}
    
    def _log_stage_end(self, stage_name: str, **kwargs):
        """Log the end of a pipeline stage"""
        stage_stats = self.execution_stats['stages'][stage_name]
        stage_stats['end_time'] = time.time()
        stage_stats['duration'] = stage_stats['end_time'] - stage_stats['start_time']
        stage_stats.update(kwargs)
        
        logger.info(f"Stage '{stage_name}' completed in {stage_stats['duration']:.2f} seconds")
        if kwargs:
            for key, value in kwargs.items():
                logger.info(f"  {key}: {value}")
    
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate input data"""
        
        self._log_stage_start("Data Loading & Validation")
        
        try:
            students_df, internships_df = load_data()
            
            # Data validation
            self._validate_data(students_df, internships_df)
            
            # Data quality checks
            quality_report = self._assess_data_quality(students_df, internships_df)
            
            self._log_stage_end("Data Loading & Validation", 
                              students=len(students_df),
                              internships=len(internships_df),
                              total_capacity=int(internships_df['capacity'].sum()),
                              data_quality_score=quality_report['overall_score'])
            
            return students_df, internships_df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _validate_data(self, students_df: pd.DataFrame, internships_df: pd.DataFrame):
        """Validate data integrity"""
        
        # Check required columns
        required_student_cols = ['id', 'name', 'email', 'gpa', 'skills', 'category']
        required_internship_cols = ['id', 'company', 'title', 'skills_required', 'capacity']
        
        for col in required_student_cols:
            if col not in students_df.columns:
                raise ValueError(f"Missing required student column: {col}")
        
        for col in required_internship_cols:
            if col not in internships_df.columns:
                raise ValueError(f"Missing required internship column: {col}")
        
        # Check for duplicates
        if students_df['id'].duplicated().any():
            raise ValueError("Duplicate student IDs found")
        
        if internships_df['id'].duplicated().any():
            raise ValueError("Duplicate internship IDs found")
        
        # Check data ranges
        if (students_df['gpa'] < 0).any() or (students_df['gpa'] > 10).any():
            logger.warning("Some GPA values are outside expected range [0, 10]")
        
        if (internships_df['capacity'] <= 0).any():
            raise ValueError("All internships must have positive capacity")
        
        logger.info("Data validation passed")
    
    def _assess_data_quality(self, students_df: pd.DataFrame, internships_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and return report"""
        
        quality_issues = []
        scores = {}
        
        # Students data quality
        student_missing_skills = students_df['skills'].isna().sum()
        if student_missing_skills > 0:
            quality_issues.append(f"{student_missing_skills} students missing skills")
        scores['student_completeness'] = 1.0 - (student_missing_skills / len(students_df))
        
        # Internships data quality  
        internship_missing_skills = internships_df['skills_required'].isna().sum()
        if internship_missing_skills > 0:
            quality_issues.append(f"{internship_missing_skills} internships missing required skills")
        scores['internship_completeness'] = 1.0 - (internship_missing_skills / len(internships_df))
        
        # Category distribution balance
        category_counts = students_df['category'].value_counts()
        category_balance = 1.0 - (category_counts.std() / category_counts.mean())
        scores['category_balance'] = max(0, category_balance)
        
        # Overall score
        overall_score = np.mean(list(scores.values()))
        
        quality_report = {
            'overall_score': round(overall_score, 3),
            'individual_scores': scores,
            'issues': quality_issues
        }
        
        if quality_issues:
            logger.warning("Data quality issues detected:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        
        return quality_report
    
    def compute_similarities(self, students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                           force_reload: bool = False, top_k: int = 50) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute semantic similarities using SentenceTransformers"""
        
        self._log_stage_start("Similarity Computation (SentenceTransformers)")
        
        try:
            # Run similarity computation with enhanced engine
            similarity_matrix, similarity_df = run_similarity(
                force_reload=force_reload, 
                top_k=top_k
            )
            
            # Quality assessment
            avg_similarity = similarity_df['similarity_score'].mean()
            min_similarity = similarity_df['similarity_score'].min()
            max_similarity = similarity_df['similarity_score'].max()
            
            # Check for potential issues
            low_quality_matches = (similarity_df['similarity_score'] < 0.3).sum()
            high_quality_matches = (similarity_df['similarity_score'] >= 0.7).sum()
            
            self._log_stage_end("Similarity Computation (SentenceTransformers)",
                              similarity_pairs=len(similarity_df),
                              avg_similarity=round(avg_similarity, 4),
                              min_similarity=round(min_similarity, 4),
                              max_similarity=round(max_similarity, 4),
                              high_quality_matches=high_quality_matches,
                              low_quality_matches=low_quality_matches)
            
            return similarity_matrix, similarity_df
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise
    
    def train_or_load_prediction_model(self, similarity_df: pd.DataFrame, 
                                     force_retrain: bool = False) -> Dict[str, Any]:
        """Train or load XGBoost success prediction model"""
        
        self._log_stage_start("Success Prediction Model (XGBoost)")
        
        try:
            model_payload = None
            
            if force_retrain:
                logger.info("Force retraining XGBoost model...")
                model_payload = train_xgb(similarity_df)
                training_metrics = model_payload.get('metrics', {})
                
                self._log_stage_end("Success Prediction Model (XGBoost)",
                                  action="trained",
                                  accuracy=round(training_metrics.get('accuracy', 0), 4),
                                  auc=round(training_metrics.get('auc', 0), 4),
                                  cv_auc=round(training_metrics.get('cv_auc_mean', 0), 4))
            else:
                try:
                    logger.info("Loading existing XGBoost model...")
                    model_payload = load_model()
                    
                    # Validate loaded model
                    if not model_payload or 'model' not in model_payload:
                        raise ValueError("Invalid model payload")
                    
                    loaded_metrics = model_payload.get('metrics', {})
                    self._log_stage_end("Success Prediction Model (XGBoost)",
                                      action="loaded",
                                      accuracy=round(loaded_metrics.get('accuracy', 0), 4),
                                      auc=round(loaded_metrics.get('auc', 0), 4))
                    
                except Exception as e:
                    logger.warning(f"Model loading failed ({e}), training new model...")
                    model_payload = train_xgb(similarity_df)
                    training_metrics = model_payload.get('metrics', {})
                    
                    self._log_stage_end("Success Prediction Model (XGBoost)",
                                      action="trained_fallback",
                                      accuracy=round(training_metrics.get('accuracy', 0), 4),
                                      auc=round(training_metrics.get('auc', 0), 4))
            
            return model_payload
            
        except Exception as e:
            logger.error(f"Prediction model setup failed: {e}")
            raise
    
    def generate_predictions(self, similarity_df: pd.DataFrame, 
                           model_payload: Dict[str, Any]) -> pd.DataFrame:
        """Generate success probability predictions"""
        
        self._log_stage_start("Success Probability Prediction")
        
        try:
            predictions_df = predict_success_prob(similarity_df, model_payload)
            
            # Analyze prediction quality
            avg_success_prob = predictions_df['success_prob_pred'].mean()
            min_success_prob = predictions_df['success_prob_pred'].min()
            max_success_prob = predictions_df['success_prob_pred'].max()
            
            high_success_predictions = (predictions_df['success_prob_pred'] >= 0.8).sum()
            low_success_predictions = (predictions_df['success_prob_pred'] < 0.4).sum()
            
            self._log_stage_end("Success Probability Prediction",
                              predictions_generated=len(predictions_df),
                              avg_success_prob=round(avg_success_prob, 4),
                              high_confidence_predictions=high_success_predictions,
                              low_confidence_predictions=low_success_predictions)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            raise
    
    def optimize_allocation(self, predictions_df: pd.DataFrame, students_df: pd.DataFrame,
                          internships_df: pd.DataFrame, top_k_optimizer: int = 30,
                          global_quotas: Dict[str, float] = None, 
                          max_time_seconds: int = 120) -> List[Dict[str, int]]:
        """Run optimization for fair allocation"""
        
        self._log_stage_start("Allocation Optimization (OR-Tools)")
        
        try:
            # Filter top candidates for optimization (reduces problem size)
            logger.info(f"Filtering to top {top_k_optimizer} candidates per student...")
            
            predictions_df = predictions_df.copy()
            predictions_df['combined_score'] = (
                predictions_df['similarity_score'] * 0.6 + 
                predictions_df['success_prob_pred'] * 0.4
            )
            
            # Keep top_k_optimizer candidates per student
            filtered_df = predictions_df.groupby('student_id').apply(
                lambda x: x.nlargest(top_k_optimizer, 'combined_score')
            ).reset_index(drop=True)
            
            logger.info(f"Optimization input: {len(filtered_df)} candidate pairs")
            
            # Set default quotas if not provided
            if global_quotas is None:
                global_quotas = {
                    'rural': settings.get('RURAL_QUOTA', 0.15),
                    'reserved': settings.get('RESERVED_QUOTA', 0.20), 
                    'female': settings.get('FEMALE_QUOTA', 0.30)
                }
            
            # Run optimization
            assignments = run_optimizer(
                filtered_df, students_df, internships_df,
                global_quotas=global_quotas,
                max_time_seconds=max_time_seconds
            )
            
            # Analyze optimization results
            placement_rate = len(assignments) / len(students_df) if len(students_df) > 0 else 0
            
            # Category-wise analysis
            category_analysis = self._analyze_allocation_fairness(assignments, students_df, internships_df)
            
            self._log_stage_end("Allocation Optimization (OR-Tools)",
                              assignments_found=len(assignments),
                              placement_rate=round(placement_rate, 4),
                              fairness_score=round(category_analysis['fairness_score'], 4))
            
            return assignments
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _analyze_allocation_fairness(self, assignments: List[Dict], 
                                   students_df: pd.DataFrame, 
                                   internships_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fairness of allocations across categories"""
        
        if not assignments:
            return {'fairness_score': 0.0, 'category_rates': {}}
        
        allocated_students = set(a['student_id'] for a in assignments)
        category_rates = {}
        
        for category in students_df['category'].unique():
            category_students = students_df[students_df['category'] == category]
            allocated_in_category = sum(1 for sid in category_students['id'] if sid in allocated_students)
            
            rate = allocated_in_category / len(category_students) if len(category_students) > 0 else 0
            category_rates[category] = rate
        
        # Calculate fairness score (1 - coefficient of variation)
        if len(category_rates) > 1:
            rates = list(category_rates.values())
            fairness_score = 1.0 - (np.std(rates) / np.mean(rates)) if np.mean(rates) > 0 else 0
        else:
            fairness_score = 1.0
        
        return {
            'fairness_score': max(0, fairness_score),
            'category_rates': category_rates
        }
    
    def validate_and_save_results(self, assignments: List[Dict], students_df: pd.DataFrame,
                                internships_df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate assignments and save comprehensive results"""
        
        self._log_stage_start("Validation & Results Generation")
        
        try:
            # Validate assignments
            validation_result = self._validate_assignments(assignments, students_df, internships_df)
            
            if not validation_result['valid']:
                logger.error("Assignment validation failed:")
                for error in validation_result['errors']:
                    logger.error(f"  - {error}")
                raise ValueError("Invalid assignments generated")
            
            # Generate comprehensive results
            results_data = self._prepare_detailed_results(assignments, students_df, internships_df, predictions_df)
            
            # Save results in multiple formats
            output_files = self._save_results(results_data, students_df, internships_df, assignments)
            
            # Generate analytics report
            analytics_report = generate_allocation_report(students_df, internships_df, assignments)
            
            self._log_stage_end("Validation & Results Generation",
                              validation_passed=validation_result['valid'],
                              files_saved=len(output_files),
                              analytics_generated=True)
            
            return {
                'assignments': assignments,
                'results_data': results_data,
                'output_files': output_files,
                'analytics': analytics_report,
                'validation': validation_result
            }
            
        except Exception as e:
            logger.error(f"Results validation and saving failed: {e}")
            raise
    
    def _validate_assignments(self, assignments: List[Dict], students_df: pd.DataFrame, 
                            internships_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive assignment validation"""
        
        errors = []
        warnings = []
        
        if not assignments:
            errors.append("No assignments found")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        assignments_df = pd.DataFrame(assignments)
        
        # Check for duplicate student assignments
        duplicate_students = assignments_df['student_id'].duplicated()
        if duplicate_students.any():
            duplicate_ids = assignments_df[duplicate_students]['student_id'].tolist()
            errors.append(f"Students assigned multiple times: {duplicate_ids}")
        
        # Check capacity constraints
        capacity_violations = []
        internship_capacities = internships_df.set_index('id')['capacity'].to_dict()
        assignment_counts = assignments_df['internship_id'].value_counts().to_dict()
        
        for internship_id, count in assignment_counts.items():
            capacity = internship_capacities.get(internship_id, 0)
            if count > capacity:
                capacity_violations.append(f"Internship {internship_id}: {count} > {capacity}")
        
        if capacity_violations:
            errors.append(f"Capacity violations: {capacity_violations}")
        
        # Check if all referenced students and internships exist
        valid_student_ids = set(students_df['id'])
        valid_internship_ids = set(internships_df['id'])
        
        invalid_students = set(assignments_df['student_id']) - valid_student_ids
        invalid_internships = set(assignments_df['internship_id']) - valid_internship_ids
        
        if invalid_students:
            errors.append(f"Invalid student IDs: {list(invalid_students)}")
        
        if invalid_internships:
            errors.append(f"Invalid internship IDs: {list(invalid_internships)}")
        
        # Warnings for low placement rate
        placement_rate = len(assignments) / len(students_df)
        if placement_rate < 0.5:
            warnings.append(f"Low placement rate: {placement_rate:.2%}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _prepare_detailed_results(self, assignments: List[Dict], students_df: pd.DataFrame,
                                internships_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare detailed results with all relevant information"""
        
        students_lookup = students_df.set_index('id').to_dict('index')
        internships_lookup = internships_df.set_index('id').to_dict('index') 
        
        # Create predictions lookup
        pred_lookup = {}
        for _, row in predictions_df.iterrows():
            key = (int(row['student_id']), int(row['internship_id']))
            pred_lookup[key] = {
                'similarity_score': float(row['similarity_score']),
                'success_prob': float(row['success_prob_pred'])
            }
        
        # Build detailed results
        results = []
        for assignment in assignments:
            student_id = assignment['student_id']
            internship_id = assignment['internship_id']
            
            student_data = students_lookup.get(student_id, {})
            internship_data = internships_lookup.get(internship_id, {})
            pred_data = pred_lookup.get((student_id, internship_id), {})
            
            results.append({
                'student_id': student_id,
                'student_name': student_data.get('name', 'Unknown'),
                'student_email': student_data.get('email', ''),
                'student_gpa': student_data.get('gpa', 0),
                'student_skills': student_data.get('skills', ''),
                'student_category': student_data.get('category', 'general'),
                'student_location': student_data.get('location', ''),
                'student_experience': student_data.get('past_internships', 0),
                
                'internship_id': internship_id,
                'company': internship_data.get('company', 'Unknown'),
                'title': internship_data.get('title', 'Unknown'),
                'required_skills': internship_data.get('skills_required', ''),
                'internship_location': internship_data.get('location', ''),
                'internship_capacity': internship_data.get('capacity', 0),
                'stipend': internship_data.get('stipend', 0),
                'duration_months': internship_data.get('duration_months', 3),
                
                'similarity_score': pred_data.get('similarity_score', 0),
                'success_probability': pred_data.get('success_prob', 0),
                
                'allocation_timestamp': pd.Timestamp.now()
            })
        
        return pd.DataFrame(results)
    
    def _save_results(self, results_df: pd.DataFrame, students_df: pd.DataFrame,
                     internships_df: pd.DataFrame, assignments: List[Dict]) -> List[str]:
        """Save results in multiple formats"""
        
        output_files = []
        
        # Save detailed allocations
        detailed_path = "data/allocations_detailed.csv"
        results_df.to_csv(detailed_path, index=False)
        output_files.append(detailed_path)
        
        # Save simple allocations (backward compatibility)
        simple_df = results_df[['student_id', 'student_name', 'student_category', 
                              'internship_id', 'company', 'title', 
                              'similarity_score', 'success_probability']].copy()
        simple_path = "data/allocations.csv"
        simple_df.to_csv(simple_path, index=False)
        output_files.append(simple_path)
        
        # Save summary statistics
        summary = self._generate_summary_statistics(results_df, students_df, internships_df)
        summary_path = "data/allocation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_files.append(summary_path)
        
        return output_files
    
    def _generate_summary_statistics(self, results_df: pd.DataFrame, students_df: pd.DataFrame,
                                   internships_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        total_students = len(students_df)
        total_internships = len(internships_df)
        total_capacity = int(internships_df['capacity'].sum())
        total_allocated = len(results_df)
        
        # Category breakdown
        category_stats = {}
        allocated_students = set(results_df['student_id'])
        
        for category in students_df['category'].unique():
            category_students = students_df[students_df['category'] == category]
            allocated_in_category = sum(1 for sid in category_students['id'] if sid in allocated_students)
            
            category_stats[category] = {
                'total_students': len(category_students),
                'allocated': allocated_in_category,
                'placement_rate': allocated_in_category / len(category_students) if len(category_students) > 0 else 0
            }
        
        # Quality metrics
        if len(results_df) > 0:
            avg_similarity = results_df['similarity_score'].mean()
            avg_success_prob = results_df['success_probability'].mean()
            high_quality_matches = (results_df['similarity_score'] >= 0.8).sum()
        else:
            avg_similarity = avg_success_prob = high_quality_matches = 0
        
        # Company distribution
        company_distribution = results_df['company'].value_counts().to_dict() if len(results_df) > 0 else {}
        
        return {
            'execution_timestamp': pd.Timestamp.now().isoformat(),
            'overview': {
                'total_students': total_students,
                'total_internships': total_internships,
                'total_capacity': total_capacity,
                'total_allocated': total_allocated,
                'overall_placement_rate': total_allocated / total_students if total_students > 0 else 0,
                'capacity_utilization': total_allocated / total_capacity if total_capacity > 0 else 0
            },
            'category_breakdown': category_stats,
            'quality_metrics': {
                'avg_similarity_score': round(avg_similarity, 4),
                'avg_success_probability': round(avg_success_prob, 4),
                'high_quality_matches': int(high_quality_matches),
                'high_quality_percentage': round(high_quality_matches / len(results_df) * 100, 2) if len(results_df) > 0 else 0
            },
            'company_distribution': company_distribution,
            'execution_stats': self.execution_stats
        }
    
    def run_complete_pipeline(self, force_embeddings: bool = False, train_model: bool = False,
                            top_k_similarity: int = 50, top_k_optimizer: int = 30,
                            rural_quota: float = 0.15, reserved_quota: float = 0.20,
                            female_quota: float = 0.30, max_optimization_time: int = 120) -> Dict[str, Any]:
        """Run the complete allocation pipeline"""
        
        logger.info("Starting Internship Allocation Pipeline")
        logger.info(f"Parameters: embeddings_reload={force_embeddings}, retrain_model={train_model}")
        logger.info(f"Top-K: similarity={top_k_similarity}, optimizer={top_k_optimizer}")
        logger.info(f"Quotas: rural={rural_quota}, reserved={reserved_quota}, female={female_quota}")
        
        self.execution_stats['start_time'] = time.time()
        
        try:
            # Stage 1: Load and validate data
            students_df, internships_df = self.load_and_validate_data()
            
            # Stage 2: Compute similarities
            similarity_matrix, similarity_df = self.compute_similarities(
                students_df, internships_df, force_embeddings, top_k_similarity
            )
            
            # Stage 3: Train/load prediction model
            model_payload = self.train_or_load_prediction_model(similarity_df, train_model)
            
            # Stage 4: Generate predictions
            predictions_df = self.generate_predictions(similarity_df, model_payload)
            
            # Stage 5: Run optimization
            global_quotas = {
                'rural': rural_quota,
                'reserved': reserved_quota, 
                'female': female_quota
            }
            
            assignments = self.optimize_allocation(
                predictions_df, students_df, internships_df,
                top_k_optimizer, global_quotas, max_optimization_time
            )
            
            # Stage 6: Validate and save results
            final_results = self.validate_and_save_results(
                assignments, students_df, internships_df, predictions_df
            )
            
            # Final statistics
            self.execution_stats['end_time'] = time.time()
            self.execution_stats['total_time'] = self.execution_stats['end_time'] - self.execution_stats['start_time']
            self.execution_stats['success'] = True
            
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"⏱️  Total execution time: {self.execution_stats['total_time']:.2f} seconds")
            logger.info(f"📊 Allocated {len(assignments)} students out of {len(students_df)}")
            logger.info(f"📈 Placement rate: {len(assignments)/len(students_df):.2%}")
            
            return final_results
            
        except Exception as e:
            self.execution_stats['success'] = False
            self.execution_stats['error'] = str(e)
            
            logger.error("❌ Pipeline failed!")
            logger.error(f"Error: {e}")
            
            raise

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Complete Internship Allocation Engine Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python runPython.py
  
  # Full pipeline with retraining
  python runPython.py --train-model --force-embeddings
  
  # Custom parameters
  python runPython.py --top-k-sim 40 --top-k-opt 25 --quota-rural 0.20
  
  # Quick test run
  python runPython.py --top-k-sim 10 --max-time 30
        """
    )
    
    # Core pipeline options
    parser.add_argument("--force-embeddings", action="store_true",
                       help="Force recomputation of embeddings (bypasses cache)")
    parser.add_argument("--train-model", action="store_true", 
                       help="Train new XGBoost model (instead of loading existing)")
    
    # Algorithm parameters
    parser.add_argument("--top-k-sim", type=int, default=50,
                       help="Top-K candidates per student for similarity stage (default: 50)")
    parser.add_argument("--top-k-opt", type=int, default=30,
                       help="Top-K candidates per student for optimization stage (default: 30)")
    
    # Diversity quotas
    parser.add_argument("--quota-rural", type=float, default=0.15,
                       help="Rural quota fraction (default: 0.15)")
    parser.add_argument("--quota-reserved", type=float, default=0.20,
                       help="Reserved category quota fraction (default: 0.20)")
    parser.add_argument("--quota-female", type=float, default=0.30,
                       help="Female quota fraction (default: 0.30)")
    
    # Performance settings
    parser.add_argument("--max-time", type=int, default=120,
                       help="Maximum optimization time in seconds (default: 120)")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level (default: INFO)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output (errors only)")
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    if args.quiet:
        log_level = "ERROR"
    else:
        log_level = args.log_level
    
    logger = setup_logging(log_level)
    
    # Create and run pipeline
    try:
        pipeline = AllocationPipeline()
        
        results = pipeline.run_complete_pipeline(
            force_embeddings=args.force_embeddings,
            train_model=args.train_model,
            top_k_similarity=args.top_k_sim,
            top_k_optimizer=args.top_k_opt,
            rural_quota=args.quota_rural,
            reserved_quota=args.quota_reserved,
            female_quota=args.quota_female,
            max_optimization_time=args.max_time
        )
        
        # Print summary
        if not args.quiet:
            summary_path = "data/allocation_summary.json"
            if os.path.exists(summary_path):
                print(f"\n📋 Detailed summary saved to: {summary_path}")
            
            print(f"\n📁 Output files:")
            for file_path in results.get('output_files', []):
                print(f"  - {file_path}")
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
