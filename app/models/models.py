"""Database models for the Internship Allocation System"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Student(Base):
    """Student model"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    gpa = Column(Float, nullable=False)
    skills = Column(Text, nullable=False)  # Comma-separated skills
    education = Column(String(100), nullable=False)
    location = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)  # general, rural, reserved, female
    past_internships = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to allocations
    allocations = relationship("Allocation", back_populates="student")

class Internship(Base):
    """Internship model"""
    __tablename__ = "internships"
    
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String(100), nullable=False, index=True)
    title = Column(String(200), nullable=False, index=True)
    skills_required = Column(Text, nullable=False)  # Comma-separated skills
    capacity = Column(Integer, nullable=False)
    location = Column(String(100), nullable=False)
    reserved_for = Column(String(50), default="none")  # none, rural, reserved, female
    description = Column(Text)
    duration_months = Column(Integer, default=3)
    stipend = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to allocations
    allocations = relationship("Allocation", back_populates="internship")

class Allocation(Base):
    """Allocation model - represents student-internship assignments"""
    __tablename__ = "allocations"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    internship_id = Column(Integer, ForeignKey("internships.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    success_probability = Column(Float, nullable=False)
    allocation_round = Column(Integer, default=1)  # For multiple allocation rounds
    status = Column(String(50), default="allocated")  # allocated, accepted, rejected
    allocated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    student = relationship("Student", back_populates="allocations")
    internship = relationship("Internship", back_populates="allocations")

class SimilarityCache(Base):
    """Cache for similarity computations"""
    __tablename__ = "similarity_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, nullable=False)
    internship_id = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

class AnalyticsLog(Base):
    """Analytics and metrics logging"""
    __tablename__ = "analytics_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    allocation_round = Column(Integer, nullable=False)
    total_students = Column(Integer, nullable=False)
    total_internships = Column(Integer, nullable=False)
    total_allocated = Column(Integer, nullable=False)
    placement_rate = Column(Float, nullable=False)
    avg_similarity = Column(Float, nullable=False)
    avg_success_prob = Column(Float, nullable=False)
    metrics_json = Column(Text)  # JSON string of detailed metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now())
