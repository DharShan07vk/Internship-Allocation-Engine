"""Analytics and visualization utilities"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
import json
import logging

from app.models.models import Student, Internship, Allocation

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Comprehensive analytics engine for internship allocation system"""
    
    def __init__(self):
        self.students_df = None
        self.internships_df = None
        self.allocations_df = None
        
    def load_data(self, students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                  allocations: List[Dict] = None):
        """Load data for analysis"""
        self.students_df = students_df.copy()
        self.internships_df = internships_df.copy()
        
        if allocations:
            self.allocations_df = pd.DataFrame(allocations)
        else:
            self.allocations_df = pd.DataFrame()
    
    def compute_placement_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive placement metrics"""
        
        if self.allocations_df.empty:
            return self._empty_metrics()
        
        total_students = len(self.students_df)
        total_internships = len(self.internships_df)
        total_allocated = len(self.allocations_df)
        
        # Overall placement rate
        placement_rate = total_allocated / total_students if total_students > 0 else 0.0
        
        # Category-wise placement
        allocated_students = set(self.allocations_df['student_id'])
        category_metrics = {}
        
        for category in self.students_df['category'].unique():
            category_students = self.students_df[self.students_df['category'] == category]
            allocated_in_category = sum(1 for sid in category_students['id'] if sid in allocated_students)
            
            category_metrics[category] = {
                'total': len(category_students),
                'allocated': allocated_in_category,
                'placement_rate': allocated_in_category / len(category_students) if len(category_students) > 0 else 0.0
            }
        
        # Capacity utilization
        capacity_utilization = self._compute_capacity_utilization()
        
        # Quality metrics
        quality_metrics = self._compute_quality_metrics()
        
        return {
            'overview': {
                'total_students': total_students,
                'total_internships': total_internships,
                'total_capacity': int(self.internships_df['capacity'].sum()),
                'total_allocated': total_allocated,
                'overall_placement_rate': round(placement_rate, 4)
            },
            'category_breakdown': category_metrics,
            'capacity_utilization': capacity_utilization,
            'quality_metrics': quality_metrics
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no allocations exist"""
        return {
            'overview': {
                'total_students': len(self.students_df) if self.students_df is not None else 0,
                'total_internships': len(self.internships_df) if self.internships_df is not None else 0,
                'total_capacity': int(self.internships_df['capacity'].sum()) if self.internships_df is not None else 0,
                'total_allocated': 0,
                'overall_placement_rate': 0.0
            },
            'category_breakdown': {},
            'capacity_utilization': {},
            'quality_metrics': {}
        }
    
    def _compute_capacity_utilization(self) -> Dict[str, Any]:
        """Compute capacity utilization metrics"""
        
        if self.allocations_df.empty:
            return {}
        
        # Count allocations per internship
        allocation_counts = self.allocations_df['internship_id'].value_counts().to_dict()
        
        utilization_data = []
        for _, internship in self.internships_df.iterrows():
            internship_id = internship['id']
            capacity = internship['capacity']
            allocated = allocation_counts.get(internship_id, 0)
            utilization = allocated / capacity if capacity > 0 else 0
            
            utilization_data.append({
                'internship_id': internship_id,
                'company': internship['company'],
                'capacity': capacity,
                'allocated': allocated,
                'utilization_rate': utilization
            })
        
        utilization_df = pd.DataFrame(utilization_data)
        
        return {
            'average_utilization': round(utilization_df['utilization_rate'].mean(), 4),
            'full_capacity_count': sum(1 for rate in utilization_df['utilization_rate'] if rate >= 1.0),
            'underutilized_count': sum(1 for rate in utilization_df['utilization_rate'] if rate < 0.5),
            'utilization_distribution': utilization_df.groupby(
                pd.cut(utilization_df['utilization_rate'], bins=[0, 0.5, 0.8, 1.0, float('inf')], 
                       labels=['<50%', '50-80%', '80-100%', '>100%'])
            ).size().to_dict()
        }
    
    def _compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute allocation quality metrics"""
        
        if self.allocations_df.empty or 'similarity_score' not in self.allocations_df.columns:
            return {}
        
        similarity_scores = self.allocations_df['similarity_score']
        success_probs = self.allocations_df.get('success_probability', pd.Series([0.5] * len(self.allocations_df)))
        
        return {
            'avg_similarity_score': round(similarity_scores.mean(), 4),
            'min_similarity_score': round(similarity_scores.min(), 4),
            'max_similarity_score': round(similarity_scores.max(), 4),
            'similarity_std': round(similarity_scores.std(), 4),
            'avg_success_probability': round(success_probs.mean(), 4),
            'high_quality_matches': sum(1 for score in similarity_scores if score >= 0.8),
            'low_quality_matches': sum(1 for score in similarity_scores if score < 0.5)
        }
    
    def generate_visualizations(self, save_path: str = "analytics/") -> Dict[str, str]:
        """Generate comprehensive visualizations"""
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        viz_files = {}
        
        # 1. Placement rate by category
        viz_files['placement_by_category'] = self._plot_placement_by_category(save_path)
        
        # 2. Similarity score distribution
        if not self.allocations_df.empty and 'similarity_score' in self.allocations_df.columns:
            viz_files['similarity_distribution'] = self._plot_similarity_distribution(save_path)
        
        # 3. Capacity utilization
        viz_files['capacity_utilization'] = self._plot_capacity_utilization(save_path)
        
        # 4. Top companies by allocations
        viz_files['top_companies'] = self._plot_top_companies(save_path)
        
        # 5. Skill demand analysis
        viz_files['skill_demand'] = self._plot_skill_demand(save_path)
        
        return viz_files
    
    def _plot_placement_by_category(self, save_path: str) -> str:
        """Plot placement rates by student category"""
        
        if self.allocations_df.empty:
            return ""
        
        allocated_students = set(self.allocations_df['student_id'])
        
        categories = []
        total_counts = []
        allocated_counts = []
        placement_rates = []
        
        for category in self.students_df['category'].unique():
            category_students = self.students_df[self.students_df['category'] == category]
            allocated_in_category = sum(1 for sid in category_students['id'] if sid in allocated_students)
            
            categories.append(category)
            total_counts.append(len(category_students))
            allocated_counts.append(allocated_in_category)
            placement_rates.append(allocated_in_category / len(category_students) if len(category_students) > 0 else 0)
        
        # Create plotly bar chart
        fig = go.Figure(data=[
            go.Bar(name='Total Students', x=categories, y=total_counts, opacity=0.7),
            go.Bar(name='Allocated Students', x=categories, y=allocated_counts, opacity=0.9)
        ])
        
        fig.update_layout(
            title='Student Placement by Category',
            xaxis_title='Category',
            yaxis_title='Number of Students',
            barmode='group'
        )
        
        file_path = f"{save_path}placement_by_category.html"
        fig.write_html(file_path)
        
        return file_path
    
    def _plot_similarity_distribution(self, save_path: str) -> str:
        """Plot distribution of similarity scores"""
        
        if self.allocations_df.empty or 'similarity_score' not in self.allocations_df.columns:
            return ""
        
        similarity_scores = self.allocations_df['similarity_score']
        
        fig = go.Figure(data=[go.Histogram(x=similarity_scores, nbinsx=30)])
        fig.update_layout(
            title='Distribution of Similarity Scores',
            xaxis_title='Similarity Score',
            yaxis_title='Frequency'
        )
        
        file_path = f"{save_path}similarity_distribution.html"
        fig.write_html(file_path)
        
        return file_path
    
    def _plot_capacity_utilization(self, save_path: str) -> str:
        """Plot capacity utilization"""
        
        if self.allocations_df.empty:
            return ""
        
        allocation_counts = self.allocations_df['internship_id'].value_counts().to_dict()
        
        companies = []
        utilization_rates = []
        
        for _, internship in self.internships_df.iterrows():
            internship_id = internship['id']
            capacity = internship['capacity']
            allocated = allocation_counts.get(internship_id, 0)
            utilization = allocated / capacity if capacity > 0 else 0
            
            companies.append(f"{internship['company'][:20]}")
            utilization_rates.append(utilization)
        
        fig = go.Figure(data=[go.Bar(x=companies, y=utilization_rates)])
        fig.update_layout(
            title='Capacity Utilization by Company',
            xaxis_title='Company',
            yaxis_title='Utilization Rate',
            xaxis_tickangle=-45
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="100% Capacity")
        
        file_path = f"{save_path}capacity_utilization.html"
        fig.write_html(file_path)
        
        return file_path
    
    def _plot_top_companies(self, save_path: str) -> str:
        """Plot top companies by number of allocations"""
        
        if self.allocations_df.empty:
            return ""
        
        # Get company names from internship data
        internship_lookup = self.internships_df.set_index('id')['company'].to_dict()
        
        # Count allocations per company
        company_counts = {}
        for _, allocation in self.allocations_df.iterrows():
            internship_id = allocation['internship_id']
            company = internship_lookup.get(internship_id, 'Unknown')
            company_counts[company] = company_counts.get(company, 0) + 1
        
        # Get top 10 companies
        top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        companies, counts = zip(*top_companies) if top_companies else ([], [])
        
        fig = go.Figure(data=[go.Bar(x=list(companies), y=list(counts))])
        fig.update_layout(
            title='Top 10 Companies by Number of Allocations',
            xaxis_title='Company',
            yaxis_title='Number of Allocations',
            xaxis_tickangle=-45
        )
        
        file_path = f"{save_path}top_companies.html"
        fig.write_html(file_path)
        
        return file_path
    
    def _plot_skill_demand(self, save_path: str) -> str:
        """Plot skill demand analysis"""
        
        skill_counts = {}
        
        for _, internship in self.internships_df.iterrows():
            skills = str(internship['skills_required']).lower().split(',')
            for skill in skills:
                skill = skill.strip()
                if skill and len(skill) > 2:  # Filter out very short strings
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Get top 15 skills
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        skills, counts = zip(*top_skills) if top_skills else ([], [])
        
        fig = go.Figure(data=[go.Bar(x=list(counts), y=list(skills), orientation='h')])
        fig.update_layout(
            title='Top Skills in Demand',
            xaxis_title='Number of Internships Requiring Skill',
            yaxis_title='Skill',
            height=600
        )
        
        file_path = f"{save_path}skill_demand.html"
        fig.write_html(file_path)
        
        return file_path
    
    def export_report(self, file_path: str = "analytics/allocation_report.json"):
        """Export comprehensive analytics report"""
        
        metrics = self.compute_placement_metrics()
        
        # Add additional insights
        insights = self._generate_insights(metrics)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'insights': insights,
            'summary': {
                'total_processed': len(self.students_df) if self.students_df is not None else 0,
                'success_rate': metrics['overview']['overall_placement_rate'],
                'quality_score': metrics.get('quality_metrics', {}).get('avg_similarity_score', 0)
            }
        }
        
        # Save report
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analytics report saved to {file_path}")
        return report
    
    def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from metrics"""
        
        insights = []
        
        overview = metrics.get('overview', {})
        category_breakdown = metrics.get('category_breakdown', {})
        quality_metrics = metrics.get('quality_metrics', {})
        
        # Placement rate insights
        placement_rate = overview.get('overall_placement_rate', 0)
        if placement_rate < 0.6:
            insights.append("Overall placement rate is below 60%. Consider increasing internship capacity or improving matching algorithms.")
        elif placement_rate > 0.9:
            insights.append("Excellent placement rate achieved! The system is performing very well.")
        
        # Category fairness insights
        if category_breakdown:
            rates = [cat['placement_rate'] for cat in category_breakdown.values()]
            if max(rates) - min(rates) > 0.2:
                insights.append("Significant disparity in placement rates across categories. Review diversity quotas and constraints.")
        
        # Quality insights
        if quality_metrics:
            avg_similarity = quality_metrics.get('avg_similarity_score', 0)
            if avg_similarity < 0.5:
                insights.append("Average similarity score is low. Consider improving skill matching or expanding the candidate pool.")
            elif avg_similarity > 0.8:
                insights.append("High quality matches achieved with excellent similarity scores.")
        
        # Capacity insights
        capacity_util = metrics.get('capacity_utilization', {})
        if capacity_util:
            avg_util = capacity_util.get('average_utilization', 0)
            if avg_util < 0.7:
                insights.append("Internship capacity is underutilized. Consider adjusting allocation constraints or increasing student pool.")
        
        return insights

# Global analytics engine
analytics_engine = AnalyticsEngine()

def generate_allocation_report(students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                             allocations: List[Dict] = None) -> Dict[str, Any]:
    """Generate comprehensive allocation report"""
    
    analytics_engine.load_data(students_df, internships_df, allocations)
    metrics = analytics_engine.compute_placement_metrics()
    visualizations = analytics_engine.generate_visualizations()
    report = analytics_engine.export_report()
    
    return {
        'metrics': metrics,
        'visualizations': visualizations,
        'report': report
    }
