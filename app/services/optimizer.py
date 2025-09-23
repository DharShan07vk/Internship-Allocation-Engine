"""Advanced Optimization Engine using OR-Tools for Fair Internship Allocation"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time
import logging

from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllocationOptimizer:
    """Advanced optimization engine for fair and efficient internship allocation"""
    
    def __init__(self):
        self.solver = None
        self.variables = {}
        self.constraints = {}
        self.objective_terms = []
    
    def create_milp_model(self, pred_df: pd.DataFrame, students_df: pd.DataFrame, 
                         internships_df: pd.DataFrame, global_quotas: Dict[str, float] = None,
                         max_time_seconds: int = 120) -> List[Dict[str, int]]:
        """Create and solve Mixed Integer Linear Programming model for allocation"""
        
        logger.info("Creating MILP optimization model...")
        
        # Initialize solver
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            logger.error("Could not create solver")
            return []
        
        # Set time limit
        self.solver.SetTimeLimit(max_time_seconds * 1000)  # Convert to milliseconds
        
        # Prepare data structures
        students = students_df.set_index('id').to_dict('index')
        internships = internships_df.set_index('id').to_dict('index')
        
        # Create lookup for predictions
        pred_lookup = {}
        for _, row in pred_df.iterrows():
            key = (int(row['student_id']), int(row['internship_id']))
            pred_lookup[key] = {
                'similarity': float(row['similarity_score']),
                'success_prob': float(row['success_prob_pred'])
            }
        
        logger.info(f"Created prediction lookup with {len(pred_lookup)} entries")
        
        # Create decision variables
        logger.info("Creating decision variables...")
        variables = {}
        for student_id in students.keys():
            for internship_id in internships.keys():
                if (student_id, internship_id) in pred_lookup:
                    var_name = f"x_{student_id}_{internship_id}"
                    variables[(student_id, internship_id)] = self.solver.BoolVar(var_name)
        
        logger.info(f"Created {len(variables)} decision variables")
        
        # Constraint 1: Each student gets at most one internship
        logger.info("Adding student assignment constraints...")
        for student_id in students.keys():
            student_vars = [variables[(student_id, internship_id)] 
                          for internship_id in internships.keys() 
                          if (student_id, internship_id) in variables]
            if student_vars:
                constraint_name = f"student_{student_id}_one_assignment"
                self.solver.Add(sum(student_vars) <= 1, constraint_name)
        
        # Constraint 2: Internship capacity constraints
        logger.info("Adding capacity constraints...")
        for internship_id, internship_data in internships.items():
            capacity = int(internship_data['capacity'])
            internship_vars = [variables[(student_id, internship_id)] 
                             for student_id in students.keys() 
                             if (student_id, internship_id) in variables]
            if internship_vars:
                constraint_name = f"internship_{internship_id}_capacity"
                self.solver.Add(sum(internship_vars) <= capacity, constraint_name)
        
        # Constraint 3: Global diversity quotas
        if global_quotas:
            logger.info("Adding diversity quota constraints...")
            total_capacity = sum(int(i['capacity']) for i in internships.values())
            
            for category, quota in global_quotas.items():
                if category in ['rural', 'reserved', 'female']:
                    category_students = [sid for sid, sdata in students.items() 
                                       if sdata.get('category') == category]
                    
                    if category_students:
                        target_allocation = int(total_capacity * quota)
                        category_vars = [variables[(student_id, internship_id)] 
                                       for student_id in category_students
                                       for internship_id in internships.keys()
                                       if (student_id, internship_id) in variables]
                        
                        if category_vars:
                            constraint_name = f"{category}_quota_min"
                            # Minimum quota (soft constraint by allowing some flexibility)
                            self.solver.Add(sum(category_vars) >= int(target_allocation * 0.8), constraint_name)
                            
                            constraint_name = f"{category}_quota_max"
                            # Maximum quota to prevent over-representation
                            self.solver.Add(sum(category_vars) <= int(target_allocation * 1.2), constraint_name)
        
        # Constraint 4: Reserved internship constraints
        logger.info("Adding reserved internship constraints...")
        for internship_id, internship_data in internships.items():
            reserved_for = internship_data.get('reserved_for', 'none')
            if reserved_for != 'none':
                # Only students from the specified category can apply
                eligible_students = [sid for sid, sdata in students.items() 
                                   if sdata.get('category') == reserved_for]
                
                # All assignments to this internship must be from eligible students
                internship_vars = [variables[(student_id, internship_id)] 
                                 for student_id in students.keys() 
                                 if (student_id, internship_id) in variables]
                
                ineligible_vars = [variables[(student_id, internship_id)] 
                                 for student_id in students.keys() 
                                 if (student_id, internship_id) in variables and student_id not in eligible_students]
                
                if ineligible_vars:
                    constraint_name = f"internship_{internship_id}_reserved_for_{reserved_for}"
                    self.solver.Add(sum(ineligible_vars) == 0, constraint_name)
        
        # Objective: Maximize weighted sum of similarity and success probability
        logger.info("Creating objective function...")
        objective_terms = []
        
        similarity_weight = 0.6
        success_weight = 0.4
        
        for (student_id, internship_id), var in variables.items():
            pred_data = pred_lookup[(student_id, internship_id)]
            similarity_score = pred_data['similarity']
            success_prob = pred_data['success_prob']
            
            # Combined score with weights
            weighted_score = (similarity_weight * similarity_score + 
                            success_weight * success_prob)
            
            objective_terms.append(weighted_score * var)
        
        # Set objective to maximize
        self.solver.Maximize(sum(objective_terms))
        
        # Solve the model
        logger.info("Solving optimization model...")
        start_time = time.time()
        status = self.solver.Solve()
        solve_time = time.time() - start_time
        
        # Process results
        assignments = []
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            logger.info(f"Solution found in {solve_time:.2f} seconds")
            logger.info(f"Objective value: {self.solver.Objective().Value():.4f}")
            
            # Extract assignments
            for (student_id, internship_id), var in variables.items():
                if var.solution_value() > 0.5:  # Binary variable is essentially 1
                    assignments.append({
                        'student_id': student_id,
                        'internship_id': internship_id
                    })
            
            logger.info(f"Found {len(assignments)} assignments")
            
        else:
            logger.error(f"Optimization failed with status: {status}")
        
        return assignments
    
    def create_cp_model(self, pred_df: pd.DataFrame, students_df: pd.DataFrame, 
                       internships_df: pd.DataFrame, global_quotas: Dict[str, float] = None,
                       max_time_seconds: int = 120) -> List[Dict[str, int]]:
        """Alternative CP-SAT model for potentially faster solving"""
        
        logger.info("Creating CP-SAT optimization model...")
        
        # Create the model
        model = cp_model.CpModel()
        
        # Prepare data
        students = students_df.set_index('id').to_dict('index')
        internships = internships_df.set_index('id').to_dict('index')
        
        pred_lookup = {}
        for _, row in pred_df.iterrows():
            key = (int(row['student_id']), int(row['internship_id']))
            pred_lookup[key] = {
                'similarity': int(row['similarity_score'] * 10000),  # Scale for integer
                'success_prob': int(row['success_prob_pred'] * 10000)
            }
        
        # Create variables
        variables = {}
        for student_id in students.keys():
            for internship_id in internships.keys():
                if (student_id, internship_id) in pred_lookup:
                    variables[(student_id, internship_id)] = model.NewBoolVar(f'x_{student_id}_{internship_id}')
        
        # Constraints
        # Each student gets at most one internship
        for student_id in students.keys():
            student_vars = [variables[(student_id, internship_id)] 
                          for internship_id in internships.keys() 
                          if (student_id, internship_id) in variables]
            if student_vars:
                model.Add(sum(student_vars) <= 1)
        
        # Capacity constraints
        for internship_id, internship_data in internships.items():
            capacity = int(internship_data['capacity'])
            internship_vars = [variables[(student_id, internship_id)] 
                             for student_id in students.keys() 
                             if (student_id, internship_id) in variables]
            if internship_vars:
                model.Add(sum(internship_vars) <= capacity)
        
        # Objective
        objective_terms = []
        for (student_id, internship_id), var in variables.items():
            pred_data = pred_lookup[(student_id, internship_id)]
            # Use scaled integer values
            weighted_score = int(0.6 * pred_data['similarity'] + 0.4 * pred_data['success_prob'])
            objective_terms.append(weighted_score * var)
        
        model.Maximize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_seconds
        
        start_time = time.time()
        status = solver.Solve(model)
        solve_time = time.time() - start_time
        
        assignments = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.info(f"CP-SAT solution found in {solve_time:.2f} seconds")
            logger.info(f"Objective value: {solver.ObjectiveValue()}")
            
            for (student_id, internship_id), var in variables.items():
                if solver.Value(var) == 1:
                    assignments.append({
                        'student_id': student_id,
                        'internship_id': internship_id
                    })
            
            logger.info(f"Found {len(assignments)} assignments")
        else:
            logger.error(f"CP-SAT optimization failed with status: {status}")
        
        return assignments
    
    def run_greedy_fallback(self, pred_df: pd.DataFrame, students_df: pd.DataFrame, 
                          internships_df: pd.DataFrame) -> List[Dict[str, int]]:
        """Greedy fallback algorithm if optimization fails"""
        
        logger.info("Running greedy fallback algorithm...")
        
        # Create combined score
        pred_df = pred_df.copy()
        pred_df['combined_score'] = (0.6 * pred_df['similarity_score'] + 
                                   0.4 * pred_df['success_prob_pred'])
        
        # Sort by combined score (descending)
        pred_df = pred_df.sort_values('combined_score', ascending=False)
        
        # Track assignments
        assigned_students = set()
        internship_capacity = internships_df.set_index('id')['capacity'].to_dict()
        internship_assigned = {iid: 0 for iid in internship_capacity.keys()}
        
        assignments = []
        
        for _, row in pred_df.iterrows():
            student_id = int(row['student_id'])
            internship_id = int(row['internship_id'])
            
            # Check if student already assigned
            if student_id in assigned_students:
                continue
            
            # Check internship capacity
            if internship_assigned[internship_id] >= internship_capacity[internship_id]:
                continue
            
            # Make assignment
            assignments.append({
                'student_id': student_id,
                'internship_id': internship_id
            })
            
            assigned_students.add(student_id)
            internship_assigned[internship_id] += 1
        
        logger.info(f"Greedy algorithm assigned {len(assignments)} students")
        return assignments

# Global optimizer instance
optimizer = AllocationOptimizer()

def run_optimizer(pred_df: pd.DataFrame, students_df: pd.DataFrame, internships_df: pd.DataFrame,
                 global_quotas: Dict[str, float] = None, max_time_seconds: int = 120,
                 use_cp_sat: bool = False) -> List[Dict[str, int]]:
    """Main optimization function with fallback strategies"""
    
    if global_quotas is None:
        global_quotas = {
            'rural': settings.get('RURAL_QUOTA', 0.15),
            'reserved': settings.get('RESERVED_QUOTA', 0.20),
            'female': settings.get('FEMALE_QUOTA', 0.30)
        }
    
    max_time_seconds = max_time_seconds or settings.get('OPTIMIZATION_TIME_LIMIT', 120)
    
    try:
        # Try primary optimization method
        if use_cp_sat or len(pred_df) > 50000:  # Use CP-SAT for large problems
            assignments = optimizer.create_cp_model(pred_df, students_df, internships_df, 
                                                  global_quotas, max_time_seconds)
        else:
            assignments = optimizer.create_milp_model(pred_df, students_df, internships_df, 
                                                    global_quotas, max_time_seconds)
        
        # If optimization fails or finds no solution, use greedy fallback
        if not assignments:
            logger.warning("Optimization found no solution, falling back to greedy algorithm")
            assignments = optimizer.run_greedy_fallback(pred_df, students_df, internships_df)
        
        return assignments
        
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        logger.info("Falling back to greedy algorithm")
        return optimizer.run_greedy_fallback(pred_df, students_df, internships_df)

if __name__ == "__main__":
    # Test the optimization engine
    from app.services.similarity import run_similarity
    from app.services.prediction import predict_success_prob, train_xgb
    
    # Get similarity data
    _, sim_df = run_similarity(force_reload=False, top_k=30)
    
    # Train prediction model and get predictions
    model_payload = train_xgb(sim_df.head(5000))
    pred_df = predict_success_prob(sim_df.head(5000), model_payload)
    
    # Load student and internship data
    from app.services.similarity import load_data
    students_df, internships_df = load_data()
    
    # Run optimization
    assignments = run_optimizer(pred_df, students_df, internships_df, 
                              global_quotas={'rural': 0.15}, max_time_seconds=60)
    
    print(f"✅ Optimization complete! Found {len(assignments)} assignments")
    print("Sample assignments:")
    for i, assignment in enumerate(assignments[:10]):
        print(f"  {i+1}. Student {assignment['student_id']} -> Internship {assignment['internship_id']}")
    
    # Calculate basic metrics
    total_students = len(students_df)
    placement_rate = len(assignments) / total_students
    print(f"\nPlacement rate: {placement_rate:.2%} ({len(assignments)}/{total_students})")
