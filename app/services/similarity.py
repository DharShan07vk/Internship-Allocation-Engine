"""Advanced Similarity Service using SentenceTransformers"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Advanced similarity engine using SentenceTransformers with caching and optimization"""
    
    def __init__(self, model_name: str = None, cache_path: str = None):
        self.model_name = model_name or settings.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
        self.cache_path = cache_path or settings.get("EMBEDDING_CACHE_PATH", "models/embeddings_cache.pkl")
        self.model = None
        self.embeddings_cache = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
    
    def _load_model(self):
        """Load SentenceTransformer model"""
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _preprocess_skills(self, skills: str) -> str:
        """Enhanced preprocessing of skill strings"""
        if pd.isna(skills) or not skills:
            return ""
        
        # Clean and normalize skills
        skills = str(skills).lower()
        # Remove extra spaces and split by common delimiters
        skill_list = []
        for delimiter in [',', ';', '|', '\n']:
            skills = skills.replace(delimiter, ',')
        
        for skill in skills.split(','):
            skill = skill.strip()
            if skill:
                # Handle common abbreviations and synonyms
                skill = self._normalize_skill(skill)
                skill_list.append(skill)
        
        return " ".join(skill_list)
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill names to handle synonyms"""
        skill_synonyms = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'cpp': 'c++',
            'db': 'database',
            'sql': 'structured query language',
            'nosql': 'non-relational database',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'devops': 'development operations',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform'
        }
        
        return skill_synonyms.get(skill, skill)
    
    def _create_enhanced_features(self, skills_text: str, additional_info: Dict = None) -> str:
        """Create enhanced feature text by combining skills with additional information"""
        feature_text = skills_text
        
        if additional_info:
            # Add education level context
            if 'education' in additional_info:
                education = additional_info['education'].lower()
                if any(term in education for term in ['m.tech', 'master', 'mca', 'mba']):
                    feature_text += " advanced degree graduate level"
                elif any(term in education for term in ['b.tech', 'bachelor', 'bca', 'b.sc']):
                    feature_text += " undergraduate bachelor degree"
            
            # Add experience context
            if 'past_internships' in additional_info:
                past_exp = additional_info['past_internships']
                if past_exp > 2:
                    feature_text += " experienced multiple internships"
                elif past_exp > 0:
                    feature_text += " some internship experience"
                else:
                    feature_text += " entry level fresher"
            
            # Add GPA context (for students)
            if 'gpa' in additional_info:
                gpa = additional_info['gpa']
                if gpa >= 8.5:
                    feature_text += " high achiever excellent academic"
                elif gpa >= 7.0:
                    feature_text += " good academic performance"
        
        return feature_text
    
    def _load_cache(self):
        """Load embeddings from cache"""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded embeddings cache with {len(self.embeddings_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load embeddings cache: {e}")
            self.embeddings_cache = {}
    
    def _save_cache(self):
        """Save embeddings to cache"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Saved embeddings cache with {len(self.embeddings_cache)} entries")
        except Exception as e:
            logger.error(f"Could not save embeddings cache: {e}")
    
    def compute_embeddings(self, students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                          force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Compute embeddings for students and internships with caching"""
        
        self._load_model()
        
        if not force_recompute:
            self._load_cache()
        
        # Prepare student texts with enhanced features
        logger.info("Preparing student feature texts...")
        student_texts = []
        for _, student in tqdm(students_df.iterrows(), total=len(students_df), desc="Processing students"):
            skills_text = self._preprocess_skills(student['skills'])
            additional_info = {
                'education': student.get('education', ''),
                'past_internships': student.get('past_internships', 0),
                'gpa': student.get('gpa', 0)
            }
            enhanced_text = self._create_enhanced_features(skills_text, additional_info)
            student_texts.append(enhanced_text)
        
        # Prepare internship texts
        logger.info("Preparing internship feature texts...")
        internship_texts = []
        for _, internship in tqdm(internships_df.iterrows(), total=len(internships_df), desc="Processing internships"):
            skills_text = self._preprocess_skills(internship['skills_required'])
            # Add job title and company context
            title_text = internship.get('title', '').lower()
            company_text = internship.get('company', '').lower()
            enhanced_text = f"{skills_text} {title_text} {company_text}"
            internship_texts.append(enhanced_text)
        
        # Compute embeddings with caching
        cache_key_students = f"students_{hash(str(student_texts))}"
        cache_key_internships = f"internships_{hash(str(internship_texts))}"
        
        if not force_recompute and cache_key_students in self.embeddings_cache:
            logger.info("Using cached student embeddings")
            student_embs = self.embeddings_cache[cache_key_students]
        else:
            logger.info("Computing student embeddings...")
            student_embs = self.model.encode(
                student_texts, 
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
            self.embeddings_cache[cache_key_students] = student_embs
        
        if not force_recompute and cache_key_internships in self.embeddings_cache:
            logger.info("Using cached internship embeddings")
            internship_embs = self.embeddings_cache[cache_key_internships]
        else:
            logger.info("Computing internship embeddings...")
            internship_embs = self.model.encode(
                internship_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
            self.embeddings_cache[cache_key_internships] = internship_embs
        
        # Save cache
        self._save_cache()
        
        return student_embs, internship_embs
    
    def compute_similarity_matrix(self, student_embs: np.ndarray, internship_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between students and internships"""
        logger.info(f"Computing similarity matrix: {student_embs.shape} x {internship_embs.shape}")
        similarity_matrix = cosine_similarity(student_embs, internship_embs)
        return similarity_matrix
    
    def build_similarity_dataframe(self, students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                                 similarity_matrix: np.ndarray, top_k: int = 50,
                                 similarity_threshold: float = 0.1) -> pd.DataFrame:
        """Build similarity dataframe with top-k matches per student"""
        
        logger.info(f"Building similarity dataframe with top_k={top_k}, threshold={similarity_threshold}")
        results = []
        
        for s_idx, student in tqdm(students_df.iterrows(), desc="Processing similarity matches"):
            similarities = similarity_matrix[s_idx]
            
            # Filter by threshold and get top-k
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            if len(valid_indices) == 0:
                # If no matches above threshold, take top 5
                valid_indices = np.argsort(similarities)[-5:]
            
            # Get top-k from valid indices
            valid_similarities = similarities[valid_indices]
            top_k_indices = np.argsort(valid_similarities)[::-1][:top_k]
            top_k_internship_indices = valid_indices[top_k_indices]
            
            for i_idx in top_k_internship_indices:
                internship = internships_df.iloc[i_idx]
                results.append({
                    'student_id': int(student['id']),
                    'student_name': student['name'],
                    'internship_id': int(internship['id']),
                    'company': internship['company'],
                    'title': internship['title'],
                    'similarity_score': round(float(similarities[i_idx]), 4),
                    'student_skills': student['skills'],
                    'required_skills': internship['skills_required']
                })
        
        similarity_df = pd.DataFrame(results)
        logger.info(f"Generated {len(similarity_df)} similarity matches")
        
        return similarity_df

# Global similarity engine instance
similarity_engine = SimilarityEngine()

def load_data(students_path: str = None, internships_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load student and internship data - supports both regular and enhanced files"""
    
    # Try enhanced files first, then fall back to regular files
    students_path = students_path or settings.get("STUDENTS_DATA_PATH", "data/students.csv")
    internships_path = internships_path or settings.get("INTERNSHIPS_DATA_PATH", "data/internships.csv")
    
    # Check for enhanced files
    enhanced_students = "data/students_enhanced.csv"
    enhanced_internships = "data/internships_enhanced.csv"
    
    if os.path.exists(enhanced_students) and students_path == "data/students.csv":
        students_path = enhanced_students
    if os.path.exists(enhanced_internships) and internships_path == "data/internships.csv":
        internships_path = enhanced_internships
    
    logger.info(f"Loading data from {students_path} and {internships_path}")
    
    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)
    
    logger.info(f"Loaded {len(students_df)} students and {len(internships_df)} internships")
    
    return students_df, internships_df

def run_similarity(force_reload: bool = False, top_k: int = 50) -> Tuple[np.ndarray, pd.DataFrame]:
    """Main function to run similarity computation"""
    
    # Load data
    students_df, internships_df = load_data()
    
    # Compute embeddings
    student_embs, internship_embs = similarity_engine.compute_embeddings(
        students_df, internships_df, force_recompute=force_reload
    )
    
    # Compute similarity matrix
    similarity_matrix = similarity_engine.compute_similarity_matrix(student_embs, internship_embs)
    
    # Build similarity dataframe
    similarity_threshold = settings.get("SIMILARITY_THRESHOLD", 0.1)
    similarity_df = similarity_engine.build_similarity_dataframe(
        students_df, internships_df, similarity_matrix, 
        top_k=top_k, similarity_threshold=similarity_threshold
    )
    
    return similarity_matrix, similarity_df

# Backward compatibility
def preprocess_skills(skills):
    return similarity_engine._preprocess_skills(skills)

def compute_embeddings(students_df, internships_df):
    return similarity_engine.compute_embeddings(students_df, internships_df)

def compute_similarity_matrix(student_embs, internship_embs):
    return similarity_engine.compute_similarity_matrix(student_embs, internship_embs)

def build_similarity_df(students_df, internships_df, sim_matrix, top_k=3):
    return similarity_engine.build_similarity_dataframe(students_df, internships_df, sim_matrix, top_k)

if __name__ == "__main__":
    # Test the similarity engine
    sim_matrix, sim_df = run_similarity(force_reload=False, top_k=10)
    
    # Save results
    output_path = "data/similarity_results.csv"
    sim_df.to_csv(output_path, index=False)
    
    print(f"✅ Similarity computation complete!")
    print(f"Generated {len(sim_df)} similarity pairs")
    print(f"Results saved to {output_path}")
    print("\nSample results:")
    print(sim_df.head(10))
