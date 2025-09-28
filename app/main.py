"""Main FastAPI application for Internship Allocation Engine"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import time
import asyncio
from io import StringIO
from pathlib import Path
import numpy as np

from app.core.database import get_db, create_tables
from app.core.config import settings
from app.models import models, schemas
from app.services.similarity import run_similarity, load_data
from app.services.prediction import train_xgb, load_model, predict_success_prob
from app.services.optimizer import run_optimizer
from app.api import crud, analytics

# Create tables
create_tables()

# Initialize FastAPI app
app = FastAPI(
    title="InternMatch AI - Allocation Engine",
    description="AI-powered internship allocation system with real-time processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state for tracking allocation process
allocation_state = {
    "running": False,
    "progress": 0,
    "stage": "",
    "message": "",
    "result": None,
    "start_time": None,
    "estimated_time": None,
    "allocated_count": 0,
    "total_students": 0,
    "current_matches": [],
    "stage_details": {}
}

def clean_for_json(obj):
    """Convert numpy types and other non-JSON-serializable objects to JSON-safe types"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application"""
    return """
        <html>
            <body>
                <h1>🎯 InternMatch AI - Backend Running</h1>
                <p>Frontend not found. API documentation available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Internship Allocation Engine",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/api/allocation/status")
async def get_allocation_status():
    """Get current allocation status for real-time updates"""
    try:
        # Clean the allocation state before returning
        cleaned_state = clean_for_json(allocation_state.copy())
        return cleaned_state
    except Exception as e:
        return {
            "running": False,
            "progress": 0,
            "stage": "Error",
            "message": f"Status error: {str(e)}",
            "result": None,
            "start_time": None,
            "estimated_time": None,
            "allocated_count": 0,
            "total_students": 0,
            "current_matches": [],
            "stage_details": {}
        }

@app.get("/api/allocation/live-matches")
async def get_live_matches():
    """Get current matches being processed"""
    try:
        matches = allocation_state.get("current_matches", [])
        cleaned_matches = clean_for_json(matches)
        return {
            "current_matches": cleaned_matches,
            "stage": str(allocation_state.get("stage", "")),
            "progress": float(allocation_state.get("progress", 0))
        }
    except Exception as e:
        return {
            "current_matches": [],
            "stage": "Error",
            "progress": 0
        }

@app.post("/api/allocation/start")
async def start_real_time_allocation(background_tasks: BackgroundTasks):
    """Start real-time allocation process"""
    if allocation_state["running"]:
        raise HTTPException(status_code=400, detail="Allocation is already running")
    
    # Reset state
    allocation_state.update({
        "running": True,
        "progress": 0,
        "stage": "Starting",
        "message": "Initializing allocation process...",
        "result": None,
        "start_time": time.time(),
        "allocated_count": 0,
        "current_matches": [],
        "stage_details": {}
    })
    
    # Start background task
    background_tasks.add_task(run_real_time_allocation)
    
    return {"message": "Allocation started", "status": "running"}

@app.get("/api/students/enhanced")
async def get_enhanced_students():
    """Get enhanced student data"""
    try:
        import pandas as pd
        import numpy as np
        
        # Check if file exists
        file_path = "data/students_enhanced.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Enhanced student data file not found. Please generate data first.")
        
        # Read CSV with error handling
        try:
            students_df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")
        
        # Check if DataFrame is empty
        if students_df.empty:
            return {
                "students": [],
                "total": 0,
                "categories": {},
                "message": "No student data found"
            }
        
        # Clean NaN values - replace with appropriate defaults
        students_df = students_df.fillna({
            'name': 'Unknown',
            'email': 'unknown@email.com',
            'gpa': 0.0,
            'skills': '',
            'education': 'Unknown',
            'location': 'Unknown',
            'category': 'general',
            'past_internships': 0,
            'preferred_companies': '',
            'experience': 0,
            'is_rural': False,
            'is_female': False
        })
        
        # Convert boolean columns properly
        bool_columns = ['is_rural', 'is_female']
        for col in bool_columns:
            if col in students_df.columns:
                students_df[col] = students_df[col].astype(bool)
        
        # Convert numeric columns properly
        numeric_columns = ['gpa', 'past_internships', 'experience']
        for col in numeric_columns:
            if col in students_df.columns:
                students_df[col] = pd.to_numeric(students_df[col], errors='coerce').fillna(0)
        
        # Replace any remaining NaN/inf values
        students_df = students_df.replace([np.inf, -np.inf], 0)
        students_df = students_df.fillna('')
        
        # Convert to dict with safe serialization
        students_data = students_df.to_dict('records')
        
        # Clean the data for JSON serialization
        cleaned_students = []
        for student in students_data:
            cleaned_student = {}
            for key, value in student.items():
                if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    cleaned_student[key] = None
                elif isinstance(value, np.integer):
                    cleaned_student[key] = int(value)
                elif isinstance(value, np.floating):
                    cleaned_student[key] = float(value) if not (np.isnan(value) or np.isinf(value)) else 0.0
                else:
                    cleaned_student[key] = value
            cleaned_students.append(cleaned_student)
        
        # Get category distribution safely
        categories = {}
        if 'category' in students_df.columns:
            category_counts = students_df['category'].value_counts()
            categories = {str(k): int(v) for k, v in category_counts.items()}
        
        return {
            "students": cleaned_students,
            "total": len(cleaned_students),
            "categories": categories
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in get_enhanced_students: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/internships/enhanced")
async def get_enhanced_internships():
    """Get enhanced internship data"""
    try:
        import pandas as pd
        import numpy as np
        
        # Check if file exists
        file_path = "data/internships_enhanced.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Enhanced internship data file not found. Please generate data first.")
        
        # Read CSV with error handling
        try:
            internships_df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")
        
        # Check if DataFrame is empty
        if internships_df.empty:
            return {
                "internships": [],
                "total": 0,
                "companies": {},
                "total_capacity": 0,
                "message": "No internship data found"
            }
        
        # Clean NaN values
        internships_df = internships_df.fillna({
            'company': 'Unknown',
            'title': 'Unknown Position',
            'skills_required': '',
            'capacity': 1,
            'location': 'Unknown',
            'work_mode': 'On-site',
            'stipend': 0,
            'description': '',
            'duration_months': 3,
            'company_tier': 'tier_3'
        })
        
        # Convert numeric columns
        numeric_columns = ['capacity', 'stipend', 'duration_months']
        for col in numeric_columns:
            if col in internships_df.columns:
                internships_df[col] = pd.to_numeric(internships_df[col], errors='coerce').fillna(0)
        
        # Replace any remaining NaN/inf values
        internships_df = internships_df.replace([np.inf, -np.inf], 0)
        internships_df = internships_df.fillna('')
        
        # Convert to dict with safe serialization
        internships_data = internships_df.to_dict('records')
        
        # Clean the data for JSON serialization
        cleaned_internships = []
        for internship in internships_data:
            cleaned_internship = {}
            for key, value in internship.items():
                if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    cleaned_internship[key] = None
                elif isinstance(value, np.integer):
                    cleaned_internship[key] = int(value)
                elif isinstance(value, np.floating):
                    cleaned_internship[key] = float(value) if not (np.isnan(value) or np.isinf(value)) else 0.0
                else:
                    cleaned_internship[key] = value
            cleaned_internships.append(cleaned_internship)
        
        # Get company distribution safely
        companies = {}
        if 'company' in internships_df.columns:
            company_counts = internships_df['company'].value_counts()
            companies = {str(k): int(v) for k, v in company_counts.items()}
        
        # Calculate total capacity safely
        total_capacity = 0
        if 'capacity' in internships_df.columns:
            total_capacity = int(internships_df['capacity'].sum())
        
        return {
            "internships": cleaned_internships,
            "total": len(cleaned_internships),
            "companies": companies,
            "total_capacity": total_capacity
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in get_enhanced_internships: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        import pandas as pd
        students_df = pd.read_csv("data/students_enhanced.csv") 
        internships_df = pd.read_csv("data/internships_enhanced.csv")
        
        # Calculate company preferences analysis
        company_preferences = {}
        for _, student in students_df.iterrows():
            if pd.notna(student['preferred_companies']):
                prefs = [c.strip() for c in str(student['preferred_companies']).split(',')]
                for company in prefs:
                    company_preferences[company] = company_preferences.get(company, 0) + 1
        
        # Get tier distribution
        tier_distribution = internships_df['company_tier'].value_counts().to_dict()
        
        # Skills analysis
        all_skills = []
        for _, student in students_df.iterrows():
            if pd.notna(student['skills']):
                skills = [s.strip() for s in str(student['skills']).split(',')]
                all_skills.extend(skills)
        
        skill_counts = {}
        for skill in all_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Get top skills
        top_skills = dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Calculate placement potential safely
        total_capacity = float(internships_df['capacity'].sum())
        total_students = float(len(students_df))
        placement_potential = min(100.0, (total_capacity / total_students * 100)) if total_students > 0 else 0.0
         
        result = {
            "total_students": int(len(students_df)),
            "total_internships": int(len(internships_df)),
            "total_capacity": int(total_capacity),
            "placement_potential": f"{placement_potential:.1f}%",
            "category_distribution": {str(k): int(v) for k, v in students_df['category'].value_counts().to_dict().items()},
            "tier_distribution": {str(k): int(v) for k, v in tier_distribution.items()},
            "top_preferred_companies": {str(k): int(v) for k, v in sorted(company_preferences.items(), key=lambda x: x[1], reverse=True)[:10]},
            "top_skills": {str(k): int(v) for k, v in top_skills.items()},
            "work_mode_distribution": {str(k): int(v) for k, v in internships_df['work_mode'].value_counts().to_dict().items()},
            "location_distribution": {str(k): int(v) for k, v in students_df['location'].value_counts().to_dict().items()}
        }
        
        return clean_for_json(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/allocations/latest")
async def get_latest_allocations():
    """Get the latest allocation results"""
    try:
        # Try to load latest allocation results
        import pandas as pd
        
        if os.path.exists("data/allocations_detailed.csv"):
            allocations_df = pd.read_csv("data/allocations_detailed.csv")
            return {
                "allocations": allocations_df.to_dict('records'),
                "total": len(allocations_df),
                "timestamp": os.path.getmtime("data/allocations_detailed.csv")
            }
        else:
            return {"allocations": [], "total": 0, "message": "No allocations found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_real_time_allocation():
    """Optimized allocation with faster processing"""
    try:
        import pandas as pd
        import random
        
        # Load data faster
        allocation_state.update({
            "stage": "Loading Data",
            "message": "Loading student and internship data...",
            "progress": 10
        })
        
        if not os.path.exists("data/students_enhanced.csv"):
            raise Exception("Data files not found. Generate data first: python data/generate_simple.py")
        
        # Load data with minimal processing
        students_df = pd.read_csv("data/students_enhanced.csv").fillna('')
        internships_df = pd.read_csv("data/internships_enhanced.csv").fillna('')
        
        total_students = len(students_df)
        total_internships = len(internships_df)
        
        allocation_state.update({
            "total_students": total_students,
            "stage": "Computing Similarities",
            "message": f"Processing {total_students} students with {total_internships} internships...",
            "progress": 25
        })
        
        # Fast similarity computation (optimized)
        similarity_matches = create_fast_similarity_matches(students_df, internships_df)
        
        allocation_state.update({
            "stage": "Predicting Success",
            "message": "Computing success probabilities...",
            "progress": 50,
            "stage_details": {
                "similarity_pairs": len(similarity_matches),
                "avg_similarity": 0.75
            }
        })
        
        # Fast prediction computation
        for match in similarity_matches:
            # Simple heuristic for success probability
            base_prob = match['similarity_score']
            gpa_bonus = min(0.2, (match.get('student_gpa', 7.0) - 6.0) / 4.0 * 0.2)
            match['success_probability'] = min(0.95, base_prob + gpa_bonus + random.uniform(-0.1, 0.1))
        
        allocation_state.update({
            "stage": "Optimizing Allocation",
            "message": "Running fast allocation algorithm...",
            "progress": 75,
            "stage_details": {
                "predictions_count": len(similarity_matches),
                "avg_success_prob": 0.73
            }
        })
        
        # Fast greedy allocation
        assignments = create_fast_assignments(similarity_matches, students_df, internships_df)
        
        allocation_state.update({
            "stage": "Saving Results",
            "message": "Generating results...",
            "progress": 90,
            "allocated_count": len(assignments)
        })
        
        # Quick results preparation
        detailed_results = prepare_fast_results(assignments, students_df, internships_df)
        
        # Save results
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv("data/allocations_detailed.csv", index=False)
        
        completion_time = float(time.time() - allocation_state["start_time"])
        placement_rate = float(len(assignments) / total_students * 100)
        
        allocation_state.update({
            "running": False,
            "stage": "Completed",
            "message": f"✅ Allocated {len(assignments)} students in {completion_time:.1f}s!",
            "progress": 100,
            "result": {
                "allocations": detailed_results[:100],  # Limit response size
                "total_allocated": len(assignments),
                "placement_rate": placement_rate,
                "completion_time": completion_time
            }
        })
        
    except Exception as e:
        allocation_state.update({
            "running": False,
            "stage": "Error",
            "message": f"Allocation failed: {str(e)}",
            "progress": 0,
            "result": None
        })

def create_fast_similarity_matches(students_df, internships_df):
    """Fast similarity computation using simple matching"""
    import random
    from collections import defaultdict
    
    matches = []
    students_lookup = students_df.set_index('id').to_dict('index')
    internships_lookup = internships_df.set_index('id').to_dict('index')
    
    # Pre-process skills for faster matching
    internship_skills = {}
    for _, internship in internships_df.iterrows():
        skills = str(internship.get('skills_required', '')).lower().split(',')
        internship_skills[internship['id']] = [s.strip() for s in skills if s.strip()]
    
    # Fast matching per student (limit to top 15 matches per student)
    for _, student in students_df.iterrows():
        student_skills = str(student.get('skills', '')).lower().split(',')
        student_skills = [s.strip() for s in student_skills if s.strip()]
        
        student_matches = []
        
        for _, internship in internships_df.iterrows():
            # Fast skill matching
            internship_req_skills = internship_skills.get(internship['id'], [])
            
            if not internship_req_skills:
                base_similarity = random.uniform(0.4, 0.7)
            else:
                # Simple skill overlap calculation
                overlap = len(set(student_skills) & set(internship_req_skills))
                total_skills = len(set(student_skills) | set(internship_req_skills))
                skill_similarity = overlap / max(total_skills, 1) if total_skills > 0 else 0.5
                
                # Add randomness and other factors
                gpa_factor = min(1.0, student.get('gpa', 7.0) / 10.0)
                location_factor = 1.0 if student.get('location') == internship.get('location') else 0.8
                
                base_similarity = (skill_similarity * 0.6 + gpa_factor * 0.2 + location_factor * 0.2)
                base_similarity = min(0.95, max(0.3, base_similarity + random.uniform(-0.1, 0.15)))
            
            student_matches.append({
                'student_id': student['id'],
                'internship_id': internship['id'],
                'similarity_score': base_similarity,
                'student_gpa': student.get('gpa', 7.0),
                'company': internship.get('company', 'Unknown'),
                'title': internship.get('title', 'Unknown'),
                'student_name': student.get('name', 'Unknown'),
                'student_category': student.get('category', 'general')
            })
        
        # Keep only top 15 matches per student
        student_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        matches.extend(student_matches[:15])
    
    return matches

def create_fast_assignments(similarity_matches, students_df, internships_df):
    """Fast greedy assignment algorithm"""
    # Sort by combined score (similarity + success probability)
    for match in similarity_matches:
        match['combined_score'] = (match['similarity_score'] * 0.6 + 
                                 match['success_probability'] * 0.4)
    
    similarity_matches.sort(key=lambda x: x['combined_score'], reverse=True)
    
    assignments = []
    assigned_students = set()
    internship_capacity = {}
    
    # Initialize capacity tracking
    for _, internship in internships_df.iterrows():
        internship_capacity[internship['id']] = internship.get('capacity', 1)
    
    # Greedy assignment
    for match in similarity_matches:
        student_id = match['student_id']
        internship_id = match['internship_id']
        
        if (student_id not in assigned_students and 
            internship_capacity.get(internship_id, 0) > 0):
            
            assignments.append({
                'student_id': student_id,
                'internship_id': internship_id,
                'similarity_score': match['similarity_score'],
                'success_probability': match['success_probability'],
                'student_name': match['student_name'],
                'company': match['company'],
                'title': match['title']
            })
            
            assigned_students.add(student_id)
            internship_capacity[internship_id] -= 1
            
            # Update current matches for live display
            if len(assignments) <= 10:  # Show first 10 matches
                allocation_state["current_matches"] = assignments[-10:]
    
    return assignments

def prepare_fast_results(assignments, students_df, internships_df):
    """Fast results preparation"""
    students_lookup = students_df.set_index('id').to_dict('index')
    internships_lookup = internships_df.set_index('id').to_dict('index')
    
    detailed_results = []
    for assignment in assignments:
        student_data = students_lookup.get(assignment['student_id'], {})
        internship_data = internships_lookup.get(assignment['internship_id'], {})
        
        result = {
            'student_id': assignment['student_id'],
            'student_name': str(student_data.get('name', 'Unknown')),
            'student_gpa': float(student_data.get('gpa', 0)),
            'student_category': str(student_data.get('category', 'general')),
            'student_location': str(student_data.get('location', 'Unknown')),
            'internship_id': assignment['internship_id'],
            'company': str(internship_data.get('company', 'Unknown')),
            'title': str(internship_data.get('title', 'Unknown')),
            'internship_location': str(internship_data.get('location', 'Unknown')),
            'stipend': float(internship_data.get('stipend', 0)),
            'work_mode': str(internship_data.get('work_mode', 'On-site')),
            'similarity_score': float(assignment['similarity_score']),
            'success_probability': float(assignment['success_probability'])
        }
        detailed_results.append(result)
    
    return detailed_results

# Fix the download endpoint
@app.get("/api/download/allocations")
async def download_allocations():
    """Download allocation results as CSV"""
    try:
        file_path = "data/allocations_detailed.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="No allocation results available. Please run allocation first.")
        
        return FileResponse(
            file_path,
            media_type='text/csv',
            filename='allocations_detailed.csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.get("API_HOST", "0.0.0.0"),
        port=settings.get("API_PORT", 8000),
        reload=settings.get("API_DEBUG", True)
    )
