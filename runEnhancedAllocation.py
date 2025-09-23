"""
Enhanced Pipeline for Real-time Allocation with Enhanced Data
============================================================

This version uses the enhanced database with company preferences
and provides real-time updates to the frontend.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.similarity import run_similarity, load_data
from app.services.prediction import train_xgb, predict_success_prob
from app.services.optimizer import run_optimizer
from app.api.analytics import generate_allocation_report

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RealTimeAllocationEngine:
    """Enhanced allocation engine with real-time updates"""
    
    def __init__(self):
        self.status = {
            "running": False,
            "progress": 0,
            "stage": "Ready",
            "message": "System ready for allocation",
            "total_students": 0,
            "allocated_count": 0,
            "current_matches": []
        }
    
    def update_status(self, **kwargs):
        """Update status and save to file for frontend polling"""
        self.status.update(kwargs)
        
        # Save status to file for frontend polling
        with open('data/allocation_status.json', 'w') as f:
            json.dump(self.status, f, indent=2)
        
        logger.info(f"Stage: {self.status.get('stage', 'Unknown')} - {self.status.get('message', '')}")
    
    def run_enhanced_allocation(self):
        """Run allocation with enhanced data and real-time updates"""
        
        try:
            self.update_status(
                running=True,
                stage="Initializing",
                message="Loading enhanced student and internship data...",
                progress=5
            )
            
            # Load enhanced data
            students_df = pd.read_csv("data/students_enhanced.csv")
            internships_df = pd.read_csv("data/internships_enhanced.csv")
            
            self.update_status(
                total_students=len(students_df),
                stage="Data Loaded",
                message=f"Loaded {len(students_df)} students and {len(internships_df)} internships",
                progress=15
            )
            
            # Stage 1: Similarity Computation
            self.update_status(
                stage="Computing Similarities",
                message="Running SentenceTransformer embeddings and similarity computation...",
                progress=25
            )
            
            similarity_matrix, similarity_df = run_similarity(force_reload=False, top_k=40)
            
            # Show sample matches
            sample_matches = []
            students_lookup = students_df.set_index('id').to_dict('index')
            internships_lookup = internships_df.set_index('id').to_dict('index')
            
            for _, row in similarity_df.head(10).iterrows():
                student_data = students_lookup.get(row['student_id'], {})
                internship_data = internships_lookup.get(row['internship_id'], {})
                
                sample_matches.append({
                    'student_id': row['student_id'],
                    'student_name': student_data.get('name', f'Student {row["student_id"]}'),
                    'internship_id': row['internship_id'],
                    'company': internship_data.get('company', 'Unknown'),
                    'title': internship_data.get('title', 'Unknown'),
                    'similarity_score': float(row['similarity_score'])
                })
            
            self.update_status(
                stage="Similarities Computed",
                message=f"Generated {len(similarity_df)} similarity pairs",
                progress=45,
                current_matches=sample_matches
            )
            
            # Stage 2: Success Prediction
            self.update_status(
                stage="Training Prediction Model",
                message="Training XGBoost model for success probability prediction...",
                progress=55
            )
            
            model_payload = train_xgb(similarity_df)
            predictions_df = predict_success_prob(similarity_df, model_payload)
            
            self.update_status(
                stage="Predictions Generated",
                message=f"Generated success probabilities for {len(predictions_df)} pairs",
                progress=70
            )
            
            # Stage 3: Optimization
            self.update_status(
                stage="Running Optimization",
                message="Executing OR-Tools optimization for fair allocation...",
                progress=80
            )
            
            assignments = run_optimizer(
                predictions_df, students_df, internships_df,
                global_quotas={'rural': 0.15, 'reserved': 0.20, 'female': 0.30},
                max_time_seconds=90
            )
            
            # Stage 4: Results Generation
            self.update_status(
                stage="Generating Results",
                message="Creating detailed allocation results...",
                progress=90
            )
            
            # Generate detailed results
            detailed_results = []
            for assignment in assignments:
                student_data = students_lookup.get(assignment['student_id'], {})
                internship_data = internships_lookup.get(assignment['internship_id'], {})
                
                # Get prediction data
                pred_row = predictions_df[
                    (predictions_df['student_id'] == assignment['student_id']) & 
                    (predictions_df['internship_id'] == assignment['internship_id'])
                ]
                
                similarity_score = float(pred_row['similarity_score'].iloc[0]) if len(pred_row) > 0 else 0
                success_prob = float(pred_row['success_prob_pred'].iloc[0]) if len(pred_row) > 0 else 0
                
                detailed_results.append({
                    'student_id': assignment['student_id'],
                    'student_name': student_data.get('name', 'Unknown'),
                    'student_gpa': student_data.get('gpa', 0),
                    'student_category': student_data.get('category', 'general'),
                    'student_skills': student_data.get('skills', ''),
                    'preferred_companies': student_data.get('preferred_companies', ''),
                    'internship_id': assignment['internship_id'],
                    'company': internship_data.get('company', 'Unknown'),
                    'title': internship_data.get('title', 'Unknown'),
                    'company_tier': internship_data.get('company_tier', 'Unknown'),
                    'work_mode': internship_data.get('work_mode', 'On-site'),
                    'location': internship_data.get('location', 'Unknown'),
                    'stipend': internship_data.get('stipend', 0),
                    'duration_months': internship_data.get('duration_months', 3),
                    'similarity_score': similarity_score,
                    'success_probability': success_prob,
                    'allocation_timestamp': pd.Timestamp.now().isoformat()
                })
            
            # Save results
            results_df = pd.DataFrame(detailed_results)
            results_df.to_csv('data/allocations_enhanced.csv', index=False)
            
            # Generate summary
            placement_rate = len(assignments) / len(students_df) * 100
            
            # Category-wise analysis
            category_stats = {}
            allocated_students = set(assignment['student_id'] for assignment in assignments)
            
            for category in students_df['category'].unique():
                category_students = students_df[students_df['category'] == category]
                allocated_in_category = sum(1 for sid in category_students['id'] if sid in allocated_students)
                category_stats[category] = {
                    'total': len(category_students),
                    'allocated': allocated_in_category,
                    'rate': allocated_in_category / len(category_students) * 100 if len(category_students) > 0 else 0
                }
            
            # Final status
            self.update_status(
                running=False,
                stage="Completed",
                message=f"Successfully allocated {len(assignments)} students ({placement_rate:.1f}% placement rate)",
                progress=100,
                allocated_count=len(assignments),
                result={
                    'total_allocated': len(assignments),
                    'placement_rate': placement_rate,
                    'category_stats': category_stats,
                    'allocations': detailed_results[:50]  # First 50 for display
                }
            )
            
            # Save comprehensive summary
            summary = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_students': len(students_df),
                'total_internships': len(internships_df),
                'total_allocated': len(assignments),
                'placement_rate': placement_rate,
                'category_breakdown': category_stats,
                'execution_time': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
            with open('data/allocation_summary_enhanced.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ Enhanced allocation completed!")
            logger.info(f"📊 Allocated {len(assignments)}/{len(students_df)} students ({placement_rate:.1f}%)")
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced allocation failed: {e}")
            self.update_status(
                running=False,
                stage="Error",
                message=f"Allocation failed: {str(e)}",
                progress=0
            )
            raise

def main():
    """Run enhanced allocation"""
    
    print("🚀 Starting Enhanced Real-time Allocation Engine")
    
    # Check if enhanced data exists
    if not os.path.exists("data/students_enhanced.csv"):
        print("❌ Enhanced data not found. Please run:")
        print("   python data/generate_enhanced.py")
        return
    
    engine = RealTimeAllocationEngine()
    engine.start_time = time.time()
    
    try:
        results = engine.run_enhanced_allocation()
        
        print(f"\n✅ Allocation completed successfully!")
        print(f"📊 Results saved to:")
        print(f"   - data/allocations_enhanced.csv")
        print(f"   - data/allocation_summary_enhanced.json")
        print(f"   - data/allocation_status.json (for frontend)")
        
    except Exception as e:
        print(f"❌ Allocation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
