from fastapi import BackgroundTasks
import random
def allocate_students_to_applied_internships():
    students_path = DATA_DIR / "students_enhanced.csv"
    internships_path = DATA_DIR / "internships_enhanced.csv"
    status_path = DATA_DIR / "allocation_status.json"
    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)
    allocations = []
    allocated_count = 0
    for _, student in students_df.iterrows():
        preferred = str(student.get("preferred_companies", "")).split(",")
        preferred = [c.strip() for c in preferred if c.strip()]
        possible = internships_df[internships_df["company"].isin(preferred)]
        if not possible.empty:
            chosen = possible.sample(1).iloc[0]
            allocations.append({
                "student_id": int(student["student_id"]),
                "student_name": student["name"],
                "student_gpa": float(student["gpa"]),
                "student_category": student["category"],
                "student_skills": student["skills"],
                "preferred_companies": student["preferred_companies"],
                "internship_id": int(chosen["internship_id"]),
                "company": chosen["company"],
                "title": chosen["title"],
                "company_tier": chosen["company_tier"],
                "work_mode": chosen["work_mode"],
                "location": chosen["location"],
                "stipend": int(chosen["stipend"]),
                "duration_months": int(chosen["duration_months"]),
            })
            allocated_count += 1
    placement_rate = 100.0 * allocated_count / len(students_df) if len(students_df) else 0
    result = {
        "total_allocated": allocated_count,
        "placement_rate": placement_rate,
        "allocations": allocations
    }
    status = {
        "running": False,
        "progress": 100,
        "stage": "Completed",
        "message": f"Successfully allocated {allocated_count} students ({placement_rate:.1f}% placement rate)",
        "total_students": len(students_df),
        "allocated_count": allocated_count,
        "current_matches": [],
        "result": result
    }
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    return status
# Start allocation endpoint
@app.post("/api/allocation/start")
async def start_allocation(background_tasks: BackgroundTasks):
    """Start allocation process (students only get internships they applied for)"""
    status = allocate_students_to_applied_internships()
    return {"status": "running", "message": "Allocation started", "result": status}
"""Simple FastAPI server for Internship Allocation Engine"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import os
from pathlib import Path

app = FastAPI(title="Internship Allocation Engine", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path("data")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)

@app.get("/api/allocation/status")
async def get_allocation_status():
    """Get current allocation status"""
    try:
        with open(DATA_DIR / "allocation_status.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "running": False,
            "progress": 0,
            "stage": "Not Started",
            "message": "Allocation not started yet",
            "total_students": 0,
            "allocated_count": 0,
            "current_matches": [],
            "result": None
        }

@app.get("/api/students")
async def get_students():
    """Get all students data"""
    try:
        df = pd.read_csv(DATA_DIR / "students_enhanced.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        return {"error": "Students data not found"}

@app.get("/api/internships") 
async def get_internships():
    """Get all internships data"""
    try:
        df = pd.read_csv(DATA_DIR / "internships_enhanced.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        return {"error": "Internships data not found"}

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Read students data
        students_df = pd.read_csv(DATA_DIR / "students_enhanced.csv")
        internships_df = pd.read_csv(DATA_DIR / "internships_enhanced.csv")
        
        # Basic stats
        stats = {
            "total_students": len(students_df),
            "total_internships": len(internships_df),
            "categories": students_df['category'].value_counts().to_dict() if 'category' in students_df.columns else {},
            "skills_distribution": {},
            "company_distribution": internships_df['company'].value_counts().to_dict() if 'company' in internships_df.columns else {},
            "avg_gpa": float(students_df['gpa'].mean()) if 'gpa' in students_df.columns else 0
        }
        
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/allocation/results")
async def get_allocation_results():
    """Get allocation results"""
    try:
        with open(DATA_DIR / "allocation_status.json", "r") as f:
            status = json.load(f)
            if status.get("result") and "allocations" in status["result"]:
                return status["result"]["allocations"]
            return []
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Internship Allocation Engine Server...")
    print("📊 Frontend available at: http://127.0.0.1:8001")
    print("📡 API docs available at: http://127.0.0.1:8001/docs")
    uvicorn.run(app, host="127.0.0.1", port=8001)
