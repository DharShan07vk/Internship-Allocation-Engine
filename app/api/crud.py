"""CRUD operations for database models"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
import pandas as pd

from app.models import models, schemas

class StudentCRUD:
    """CRUD operations for Student model"""
    
    @staticmethod
    def get_student(db: Session, student_id: int) -> Optional[models.Student]:
        """Get student by ID"""
        return db.query(models.Student).filter(models.Student.id == student_id).first()
    
    @staticmethod
    def get_students(db: Session, skip: int = 0, limit: int = 100) -> List[models.Student]:
        """Get list of students with pagination"""
        return db.query(models.Student).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_student_by_email(db: Session, email: str) -> Optional[models.Student]:
        """Get student by email"""
        return db.query(models.Student).filter(models.Student.email == email).first()
    
    @staticmethod
    def create_student(db: Session, student: schemas.StudentCreate) -> models.Student:
        """Create new student"""
        db_student = models.Student(**student.dict())
        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        return db_student
    
    @staticmethod
    def update_student(db: Session, student_id: int, student_update: schemas.StudentUpdate) -> Optional[models.Student]:
        """Update student"""
        db_student = db.query(models.Student).filter(models.Student.id == student_id).first()
        if db_student:
            update_data = student_update.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_student, key, value)
            db.commit()
            db.refresh(db_student)
        return db_student
    
    @staticmethod
    def delete_student(db: Session, student_id: int) -> bool:
        """Delete student"""
        db_student = db.query(models.Student).filter(models.Student.id == student_id).first()
        if db_student:
            db.delete(db_student)
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_students_by_category(db: Session, category: str) -> List[models.Student]:
        """Get students by category"""
        return db.query(models.Student).filter(models.Student.category == category).all()
    
    @staticmethod
    def search_students(db: Session, query: str) -> List[models.Student]:
        """Search students by name, email, or skills"""
        search_pattern = f"%{query}%"
        return db.query(models.Student).filter(
            or_(
                models.Student.name.ilike(search_pattern),
                models.Student.email.ilike(search_pattern),
                models.Student.skills.ilike(search_pattern)
            )
        ).all()
    
    @staticmethod
    def bulk_create_students(db: Session, students_data: List[Dict]) -> List[models.Student]:
        """Bulk create students"""
        db_students = []
        for student_data in students_data:
            db_student = models.Student(**student_data)
            db_students.append(db_student)
        
        db.add_all(db_students)
        db.commit()
        
        for db_student in db_students:
            db.refresh(db_student)
        
        return db_students

class InternshipCRUD:
    """CRUD operations for Internship model"""
    
    @staticmethod
    def get_internship(db: Session, internship_id: int) -> Optional[models.Internship]:
        """Get internship by ID"""
        return db.query(models.Internship).filter(models.Internship.id == internship_id).first()
    
    @staticmethod
    def get_internships(db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[models.Internship]:
        """Get list of internships with pagination"""
        query = db.query(models.Internship)
        if active_only:
            query = query.filter(models.Internship.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def create_internship(db: Session, internship: schemas.InternshipCreate) -> models.Internship:
        """Create new internship"""
        db_internship = models.Internship(**internship.dict())
        db.add(db_internship)
        db.commit()
        db.refresh(db_internship)
        return db_internship
    
    @staticmethod
    def update_internship(db: Session, internship_id: int, internship_update: schemas.InternshipUpdate) -> Optional[models.Internship]:
        """Update internship"""
        db_internship = db.query(models.Internship).filter(models.Internship.id == internship_id).first()
        if db_internship:
            update_data = internship_update.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_internship, key, value)
            db.commit()
            db.refresh(db_internship)
        return db_internship
    
    @staticmethod
    def delete_internship(db: Session, internship_id: int) -> bool:
        """Delete internship"""
        db_internship = db.query(models.Internship).filter(models.Internship.id == internship_id).first()
        if db_internship:
            db.delete(db_internship)
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_internships_by_company(db: Session, company: str) -> List[models.Internship]:
        """Get internships by company"""
        return db.query(models.Internship).filter(models.Internship.company.ilike(f"%{company}%")).all()
    
    @staticmethod
    def search_internships(db: Session, query: str) -> List[models.Internship]:
        """Search internships by company, title, or skills"""
        search_pattern = f"%{query}%"
        return db.query(models.Internship).filter(
            or_(
                models.Internship.company.ilike(search_pattern),
                models.Internship.title.ilike(search_pattern),
                models.Internship.skills_required.ilike(search_pattern),
                models.Internship.location.ilike(search_pattern)
            )
        ).all()
    
    @staticmethod
    def get_available_internships(db: Session) -> List[models.Internship]:
        """Get internships with available capacity"""
        # This would need to join with allocations to check actual availability
        return db.query(models.Internship).filter(
            and_(
                models.Internship.is_active == True,
                models.Internship.capacity > 0
            )
        ).all()
    
    @staticmethod
    def bulk_create_internships(db: Session, internships_data: List[Dict]) -> List[models.Internship]:
        """Bulk create internships"""
        db_internships = []
        for internship_data in internships_data:
            db_internship = models.Internship(**internship_data)
            db_internships.append(db_internship)
        
        db.add_all(db_internships)
        db.commit()
        
        for db_internship in db_internships:
            db.refresh(db_internship)
        
        return db_internships

class AllocationCRUD:
    """CRUD operations for Allocation model"""
    
    @staticmethod
    def get_allocation(db: Session, allocation_id: int) -> Optional[models.Allocation]:
        """Get allocation by ID"""
        return db.query(models.Allocation).filter(models.Allocation.id == allocation_id).first()
    
    @staticmethod
    def get_allocations(db: Session, skip: int = 0, limit: int = 100) -> List[models.Allocation]:
        """Get list of allocations with pagination"""
        return db.query(models.Allocation).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_student_allocation(db: Session, student_id: int) -> Optional[models.Allocation]:
        """Get allocation for a specific student"""
        return db.query(models.Allocation).filter(models.Allocation.student_id == student_id).first()
    
    @staticmethod
    def get_internship_allocations(db: Session, internship_id: int) -> List[models.Allocation]:
        """Get all allocations for a specific internship"""
        return db.query(models.Allocation).filter(models.Allocation.internship_id == internship_id).all()
    
    @staticmethod
    def create_allocation(db: Session, allocation: schemas.AllocationCreate) -> models.Allocation:
        """Create new allocation"""
        db_allocation = models.Allocation(**allocation.dict())
        db.add(db_allocation)
        db.commit()
        db.refresh(db_allocation)
        return db_allocation
    
    @staticmethod
    def delete_allocation(db: Session, allocation_id: int) -> bool:
        """Delete allocation"""
        db_allocation = db.query(models.Allocation).filter(models.Allocation.id == allocation_id).first()
        if db_allocation:
            db.delete(db_allocation)
            db.commit()
            return True
        return False
    
    @staticmethod
    def clear_all_allocations(db: Session, allocation_round: int = None) -> int:
        """Clear all allocations (optionally for specific round)"""
        query = db.query(models.Allocation)
        if allocation_round:
            query = query.filter(models.Allocation.allocation_round == allocation_round)
        
        count = query.count()
        query.delete()
        db.commit()
        return count
    
    @staticmethod
    def bulk_create_allocations(db: Session, allocations_data: List[Dict]) -> List[models.Allocation]:
        """Bulk create allocations"""
        db_allocations = []
        for allocation_data in allocations_data:
            db_allocation = models.Allocation(**allocation_data)
            db_allocations.append(db_allocation)
        
        db.add_all(db_allocations)
        db.commit()
        
        for db_allocation in db_allocations:
            db.refresh(db_allocation)
        
        return db_allocations
    
    @staticmethod
    def get_allocation_statistics(db: Session) -> Dict[str, Any]:
        """Get allocation statistics"""
        total_allocations = db.query(models.Allocation).count()
        
        # Get allocations with student and internship data
        allocations_with_details = db.query(
            models.Allocation,
            models.Student,
            models.Internship
        ).join(
            models.Student, models.Allocation.student_id == models.Student.id
        ).join(
            models.Internship, models.Allocation.internship_id == models.Internship.id
        ).all()
        
        # Calculate statistics
        if not allocations_with_details:
            return {
                'total_allocations': 0,
                'avg_similarity': 0.0,
                'avg_success_probability': 0.0,
                'category_breakdown': {},
                'company_breakdown': {}
            }
        
        similarities = []
        success_probs = []
        categories = {}
        companies = {}
        
        for allocation, student, internship in allocations_with_details:
            similarities.append(allocation.similarity_score)
            success_probs.append(allocation.success_probability)
            
            # Category breakdown
            category = student.category
            categories[category] = categories.get(category, 0) + 1
            
            # Company breakdown
            company = internship.company
            companies[company] = companies.get(company, 0) + 1
        
        return {
            'total_allocations': total_allocations,
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0,
            'avg_success_probability': sum(success_probs) / len(success_probs) if success_probs else 0.0,
            'category_breakdown': categories,
            'company_breakdown': companies
        }

class AnalyticsCRUD:
    """CRUD operations for analytics and reporting"""
    
    @staticmethod
    def log_analytics(db: Session, metrics: Dict[str, Any]) -> models.AnalyticsLog:
        """Log analytics metrics"""
        analytics_log = models.AnalyticsLog(
            allocation_round=metrics.get('allocation_round', 1),
            total_students=metrics.get('total_students', 0),
            total_internships=metrics.get('total_internships', 0),
            total_allocated=metrics.get('total_allocated', 0),
            placement_rate=metrics.get('placement_rate', 0.0),
            avg_similarity=metrics.get('avg_similarity', 0.0),
            avg_success_prob=metrics.get('avg_success_prob', 0.0),
            metrics_json=str(metrics)
        )
        
        db.add(analytics_log)
        db.commit()
        db.refresh(analytics_log)
        return analytics_log
    
    @staticmethod
    def get_analytics_history(db: Session, limit: int = 10) -> List[models.AnalyticsLog]:
        """Get analytics history"""
        return db.query(models.AnalyticsLog).order_by(
            models.AnalyticsLog.created_at.desc()
        ).limit(limit).all()
    
    @staticmethod
    def export_data_to_dataframes(db: Session) -> Dict[str, pd.DataFrame]:
        """Export all data to pandas DataFrames for analysis"""
        
        # Students
        students = db.query(models.Student).all()
        students_data = []
        for student in students:
            students_data.append({
                'id': student.id,
                'name': student.name,
                'email': student.email,
                'gpa': student.gpa,
                'skills': student.skills,
                'education': student.education,
                'location': student.location,
                'category': student.category,
                'past_internships': student.past_internships
            })
        
        # Internships
        internships = db.query(models.Internship).all()
        internships_data = []
        for internship in internships:
            internships_data.append({
                'id': internship.id,
                'company': internship.company,
                'title': internship.title,
                'skills_required': internship.skills_required,
                'capacity': internship.capacity,
                'location': internship.location,
                'reserved_for': internship.reserved_for,
                'description': internship.description,
                'duration_months': internship.duration_months,
                'stipend': internship.stipend,
                'is_active': internship.is_active
            })
        
        # Allocations
        allocations = db.query(models.Allocation).all()
        allocations_data = []
        for allocation in allocations:
            allocations_data.append({
                'id': allocation.id,
                'student_id': allocation.student_id,
                'internship_id': allocation.internship_id,
                'similarity_score': allocation.similarity_score,
                'success_probability': allocation.success_probability,
                'allocation_round': allocation.allocation_round,
                'status': allocation.status
            })
        
        return {
            'students': pd.DataFrame(students_data),
            'internships': pd.DataFrame(internships_data),
            'allocations': pd.DataFrame(allocations_data)
        }

# Create global CRUD instances
student_crud = StudentCRUD()
internship_crud = InternshipCRUD()
allocation_crud = AllocationCRUD()
analytics_crud = AnalyticsCRUD()
