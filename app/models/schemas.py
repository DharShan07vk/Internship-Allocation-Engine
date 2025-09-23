"""Pydantic schemas for API request/response models"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime

# Student Schemas
class StudentBase(BaseModel):
    name: str
    email: EmailStr
    gpa: float
    skills: str
    education: str
    location: str
    category: str
    past_internships: Optional[int] = 0

    @validator('gpa')
    def validate_gpa(cls, v):
        if not 0.0 <= v <= 10.0:
            raise ValueError('GPA must be between 0.0 and 10.0')
        return v

    @validator('category')
    def validate_category(cls, v):
        allowed_categories = ['general', 'rural', 'reserved', 'female']
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {allowed_categories}')
        return v

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    gpa: Optional[float] = None
    skills: Optional[str] = None
    education: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    past_internships: Optional[int] = None

class Student(StudentBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Internship Schemas
class InternshipBase(BaseModel):
    company: str
    title: str
    skills_required: str
    capacity: int
    location: str
    reserved_for: Optional[str] = "none"
    description: Optional[str] = None
    duration_months: Optional[int] = 3
    stipend: Optional[float] = 0.0
    is_active: Optional[bool] = True

    @validator('capacity')
    def validate_capacity(cls, v):
        if v <= 0:
            raise ValueError('Capacity must be greater than 0')
        return v

    @validator('reserved_for')
    def validate_reserved_for(cls, v):
        allowed_values = ['none', 'rural', 'reserved', 'female']
        if v not in allowed_values:
            raise ValueError(f'reserved_for must be one of: {allowed_values}')
        return v

class InternshipCreate(InternshipBase):
    pass

class InternshipUpdate(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    skills_required: Optional[str] = None
    capacity: Optional[int] = None
    location: Optional[str] = None
    reserved_for: Optional[str] = None
    description: Optional[str] = None
    duration_months: Optional[int] = None
    stipend: Optional[float] = None
    is_active: Optional[bool] = None

class Internship(InternshipBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Allocation Schemas
class AllocationBase(BaseModel):
    student_id: int
    internship_id: int
    similarity_score: float
    success_probability: float
    allocation_round: Optional[int] = 1
    status: Optional[str] = "allocated"

class AllocationCreate(AllocationBase):
    pass

class Allocation(AllocationBase):
    id: int
    allocated_at: datetime
    student: Student
    internship: Internship

    class Config:
        from_attributes = True

# API Request/Response Schemas
class AllocationRequest(BaseModel):
    force_retrain: Optional[bool] = False
    force_recompute_embeddings: Optional[bool] = False
    top_k_similarity: Optional[int] = 50
    top_k_optimizer: Optional[int] = 30
    rural_quota: Optional[float] = 0.15
    reserved_quota: Optional[float] = 0.20
    female_quota: Optional[float] = 0.30
    max_optimization_time: Optional[int] = 120

class SimilarityMatch(BaseModel):
    student_id: int
    internship_id: int
    company: str
    title: str
    similarity_score: float
    success_probability: Optional[float] = None

class StudentMatches(BaseModel):
    student_id: int
    student_name: str
    matches: List[SimilarityMatch]

class AllocationResult(BaseModel):
    student_id: int
    student_name: str
    internship_id: int
    company: str
    title: str
    similarity_score: float
    success_probability: float

class AllocationResponse(BaseModel):
    success: bool
    message: str
    total_students: int
    total_internships: int
    total_allocated: int
    placement_rate: float
    allocations: List[AllocationResult]
    execution_time: float

class AnalyticsDashboard(BaseModel):
    total_students: int
    total_internships: int
    total_allocated: int
    overall_placement_rate: float
    category_breakdown: Dict[str, Dict[str, Any]]
    avg_similarity_placed: float
    avg_similarity_unplaced: float
    avg_success_prob_placed: float
    avg_success_prob_unplaced: float
    top_companies: List[Dict[str, Any]]
    skill_demand_analysis: Dict[str, int]

class BulkUploadResponse(BaseModel):
    success: bool
    message: str
    total_processed: int
    successful: int
    failed: int
    errors: List[str] = []
