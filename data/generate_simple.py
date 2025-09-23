"""Enhanced data generation script for unique student datasets with consistent internships"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import json
import os

# Set seed based on current time for uniqueness (but you can override for reproducibility)
RANDOM_SEED = int(datetime.now().timestamp()) % 100000

def set_seed(seed=None):
    """Set random seed for reproducibility"""
    if seed is None:
        seed = RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    return seed

# Fixed internship data (consistent across runs)
FIXED_COMPANIES = [
    {"name": "Google", "tier": "tier_1", "locations": ["Bangalore", "Hyderabad", "Mumbai"]},
    {"name": "Microsoft", "tier": "tier_1", "locations": ["Bangalore", "Hyderabad"]},
    {"name": "Amazon", "tier": "tier_1", "locations": ["Bangalore", "Mumbai", "Chennai"]},
    {"name": "Meta", "tier": "tier_1", "locations": ["Bangalore", "Mumbai"]},
    {"name": "Apple", "tier": "tier_1", "locations": ["Bangalore", "Hyderabad"]},
    {"name": "Netflix", "tier": "tier_1", "locations": ["Mumbai", "Bangalore"]},
    {"name": "Adobe", "tier": "tier_1", "locations": ["Bangalore", "Noida"]},
    {"name": "Salesforce", "tier": "tier_1", "locations": ["Bangalore", "Hyderabad"]},
    
    {"name": "Flipkart", "tier": "tier_2", "locations": ["Bangalore", "Delhi", "Mumbai"]},
    {"name": "Paytm", "tier": "tier_2", "locations": ["Noida", "Mumbai", "Bangalore"]},
    {"name": "Ola", "tier": "tier_2", "locations": ["Bangalore", "Mumbai"]},
    {"name": "Swiggy", "tier": "tier_2", "locations": ["Bangalore", "Hyderabad"]},
    {"name": "Zomato", "tier": "tier_2", "locations": ["Delhi", "Mumbai", "Bangalore"]},
    {"name": "BYJU'S", "tier": "tier_2", "locations": ["Bangalore", "Mumbai"]},
    {"name": "Unacademy", "tier": "tier_2", "locations": ["Bangalore", "Delhi"]},
    {"name": "PhonePe", "tier": "tier_2", "locations": ["Bangalore", "Mumbai"]},
    {"name": "Razorpay", "tier": "tier_2", "locations": ["Bangalore", "Mumbai"]},
    {"name": "Cred", "tier": "tier_2", "locations": ["Bangalore", "Mumbai"]},
    
    {"name": "TCS", "tier": "tier_3", "locations": ["Mumbai", "Pune", "Chennai", "Bangalore", "Kolkata"]},
    {"name": "Infosys", "tier": "tier_3", "locations": ["Bangalore", "Mysore", "Pune", "Hyderabad"]},
    {"name": "Wipro", "tier": "tier_3", "locations": ["Bangalore", "Hyderabad", "Chennai"]},
    {"name": "HCL", "tier": "tier_3", "locations": ["Chennai", "Noida", "Bangalore"]},
    {"name": "Tech Mahindra", "tier": "tier_3", "locations": ["Pune", "Hyderabad", "Chennai"]},
    {"name": "Cognizant", "tier": "tier_3", "locations": ["Chennai", "Bangalore", "Pune"]},
    {"name": "Accenture", "tier": "tier_3", "locations": ["Bangalore", "Mumbai", "Hyderabad"]},
    {"name": "Capgemini", "tier": "tier_3", "locations": ["Mumbai", "Pune", "Bangalore"]},
]

FIXED_ROLES = {
    "tier_1": [
        {"title": "Software Engineer", "skills": ["Python", "Java", "JavaScript", "React", "Node.js"], "stipend_range": (80000, 150000)},
        {"title": "Frontend Developer", "skills": ["React", "JavaScript", "HTML", "CSS", "TypeScript"], "stipend_range": (70000, 120000)},
        {"title": "Backend Developer", "skills": ["Python", "Java", "Node.js", "PostgreSQL", "MongoDB"], "stipend_range": (75000, 130000)},
        {"title": "Full Stack Developer", "skills": ["React", "Node.js", "Python", "JavaScript", "MongoDB"], "stipend_range": (85000, 140000)},
        {"title": "Data Scientist", "skills": ["Python", "Machine Learning", "SQL", "Pandas", "TensorFlow"], "stipend_range": (90000, 160000)},
        {"title": "DevOps Engineer", "skills": ["AWS", "Docker", "Kubernetes", "Python", "Linux"], "stipend_range": (80000, 145000)},
        {"title": "ML Engineer", "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "AWS"], "stipend_range": (95000, 170000)},
    ],
    "tier_2": [
        {"title": "Software Developer", "skills": ["Python", "Java", "JavaScript", "React"], "stipend_range": (50000, 90000)},
        {"title": "Frontend Developer", "skills": ["React", "JavaScript", "HTML", "CSS"], "stipend_range": (45000, 80000)},
        {"title": "Backend Developer", "skills": ["Python", "Java", "Node.js", "MySQL"], "stipend_range": (50000, 85000)},
        {"title": "Mobile Developer", "skills": ["React Native", "Flutter", "Java", "Kotlin"], "stipend_range": (45000, 85000)},
        {"title": "Data Analyst", "skills": ["Python", "SQL", "Excel", "Tableau"], "stipend_range": (40000, 75000)},
        {"title": "QA Engineer", "skills": ["Selenium", "Java", "Python", "Testing"], "stipend_range": (35000, 70000)},
    ],
    "tier_3": [
        {"title": "Associate Software Engineer", "skills": ["Java", "Python", "JavaScript"], "stipend_range": (25000, 50000)},
        {"title": "Junior Developer", "skills": ["Java", "Python", "HTML", "CSS"], "stipend_range": (22000, 45000)},
        {"title": "Trainee Software Engineer", "skills": ["Java", "Python", "SQL"], "stipend_range": (20000, 40000)},
        {"title": "Junior Data Analyst", "skills": ["SQL", "Excel", "Python"], "stipend_range": (20000, 40000)},
        {"title": "Support Engineer", "skills": ["Linux", "SQL", "Python"], "stipend_range": (18000, 35000)},
    ]
}

# Dynamic student generation data
STUDENT_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Shaurya", "Atharv", "Advaith", "Arnav", "Aahan", "Aryan", "Daksh", "Darsh", "Grayson", "Harsh",
    "Anaya", "Diya", "Priya", "Ananya", "Fatima", "Anika", "Kavya", "Riya", "Saanvi", "Sara",
    "Myra", "Aditi", "Aadhya", "Keya", "Khushi", "Pihu", "Pari", "Ira", "Tara", "Nisha",
    "Rohan", "Aryan", "Kartik", "Nikhil", "Rahul", "Amit", "Suresh", "Vikram", "Ajay", "Ravi",
    "Pooja", "Sneha", "Neha", "Rekha", "Sunita", "Meena", "Geeta", "Sita", "Rita", "Lata"
]

COLLEGES = [
    "IIT Delhi", "IIT Bombay", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IIT Roorkee", "IIT Guwahati",
    "NIT Trichy", "NIT Warangal", "NIT Surathkal", "NIT Calicut", "NIT Rourkela", "NIT Allahabad",
    "BITS Pilani", "BITS Goa", "BITS Hyderabad", "VIT Vellore", "VIT Chennai", "SRM Chennai",
    "Manipal Institute", "PES University", "BMS College", "RV College", "MS Ramaiah", "PESIT",
    "IIIT Hyderabad", "IIIT Bangalore", "IIIT Delhi", "IIIT Allahabad", "Delhi University",
    "Mumbai University", "Pune University", "Anna University", "Jadavpur University"
]

LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", "Noida", "Gurgaon", "Ahmedabad"]

SKILLS_POOL = [
    "Python", "Java", "JavaScript", "C++", "React", "Node.js", "Angular", "Vue.js", "Django",
    "Flask", "Spring Boot", "HTML", "CSS", "TypeScript", "PHP", "Ruby", "Go", "Rust",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "AWS", "Azure", "GCP", "Docker",
    "Kubernetes", "Git", "Linux", "Machine Learning", "Data Science", "TensorFlow", "PyTorch",
    "Pandas", "NumPy", "Scikit-learn", "Tableau", "Power BI", "Excel", "R", "Scala",
    "Selenium", "Testing", "CI/CD", "DevOps", "React Native", "Flutter", "Android", "iOS"
]

def generate_unique_students(num_students=1000, seed=None):
    """Generate unique student dataset each time"""
    if seed:
        set_seed(seed)
    else:
        current_seed = set_seed()
        print(f"Generating unique student dataset with seed: {current_seed}")
    
    students = []
    
    for i in range(num_students):
        # Generate unique combinations
        first_name = random.choice(STUDENT_NAMES)
        last_name = random.choice(STUDENT_NAMES)
        name = f"{first_name} {last_name}"
        
        # Add randomness to make names more unique
        if random.random() < 0.3:  # 30% chance to add middle initial
            middle_initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            name = f"{first_name} {middle_initial}. {last_name}"
        
        # Generate dynamic email based on name and college
        college = random.choice(COLLEGES)
        email_variations = [
            f"{first_name.lower()}.{last_name.lower()}@student.college.edu",
            f"{first_name.lower()}{last_name.lower()}@gmail.com",
            f"{first_name.lower()}{i}@college.edu",
            f"{last_name.lower()}.{first_name.lower()}@email.com"
        ]
        email = random.choice(email_variations)
        
        # Dynamic skill generation
        num_skills = random.randint(3, 8)
        skills = random.sample(SKILLS_POOL, num_skills)
        
        # Dynamic preferences
        preferred_companies = random.sample([comp["name"] for comp in FIXED_COMPANIES], random.randint(2, 6))
        
        # Random but realistic data
        gpa = round(random.uniform(6.0, 9.8), 2)
        past_internships = random.choices([0, 1, 2, 3], weights=[0.4, 0.35, 0.2, 0.05])[0]
        experience = random.randint(0, 24)  # months
        
        # Demographics with realistic distribution
        is_female = random.random() < 0.45  # 45% female
        is_rural = random.random() < 0.25   # 25% rural
        category = random.choices(
            ['general', 'obc', 'sc', 'st'], 
            weights=[0.50, 0.27, 0.15, 0.08]
        )[0]
        
        student = {
            'id': f"STU{str(i+1).zfill(5)}",
            'name': name,
            'email': email,
            'gpa': gpa,
            'skills': ', '.join(skills),
            'education': college,
            'location': random.choice(LOCATIONS),
            'category': category,
            'past_internships': past_internships,
            'preferred_companies': ', '.join(preferred_companies),
            'experience': experience,
            'is_rural': is_rural,
            'is_female': is_female
        }
        students.append(student)
    
    return pd.DataFrame(students)

def generate_consistent_internships():
    """Generate consistent internship dataset (same every time)"""
    # Use fixed seed for internships to keep them consistent
    random.seed(42)
    np.random.seed(42)
    
    internships = []
    internship_id = 1
    
    for company in FIXED_COMPANIES:
        roles = FIXED_ROLES[company["tier"]]
        
        # Each company gets 2-4 different roles
        num_roles = random.randint(2, 4)
        selected_roles = random.sample(roles, min(num_roles, len(roles)))
        
        for role in selected_roles:
            for location in company["locations"]:
                # Generate multiple positions per role per location
                num_positions = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0]
                
                for _ in range(num_positions):
                    stipend = random.randint(role["stipend_range"][0], role["stipend_range"][1])
                    capacity = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
                    
                    # Work mode distribution
                    work_mode = random.choices(
                        ['On-site', 'Remote', 'Hybrid'], 
                        weights=[0.6, 0.25, 0.15]
                    )[0]
                    
                    duration = random.choices([3, 6, 12], weights=[0.2, 0.6, 0.2])[0]
                    
                    internship = {
                        'id': f"INT{str(internship_id).zfill(5)}",
                        'company': company["name"],
                        'title': role["title"],
                        'skills_required': ', '.join(role["skills"]),
                        'capacity': capacity,
                        'location': location,
                        'work_mode': work_mode,
                        'stipend': stipend,
                        'description': f"{role['title']} position at {company['name']} in {location}",
                        'duration_months': duration,
                        'company_tier': company["tier"]
                    }
                    internships.append(internship)
                    internship_id += 1
    
    return pd.DataFrame(internships)

def save_generation_info(student_seed, num_students):
    """Save information about the current generation"""
    info = {
        "generation_time": datetime.now().isoformat(),
        "student_seed": student_seed,
        "num_students": num_students,
        "num_internships": len(generate_consistent_internships()),
        "internship_seed": 42,  # Fixed seed for internships
        "version": "2.0"
    }
    
    with open('generation_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return info

def main():
    """Main function to generate enhanced datasets"""
    print("🚀 Enhanced Dataset Generation Started")
    print("=" * 50)
    
    # Configuration
    NUM_STUDENTS = 1000
    CUSTOM_SEED = None  # Set to a number for reproducible students, None for unique
    
    # Generate datasets
    print("📊 Generating unique student dataset...")
    students_df = generate_unique_students(NUM_STUDENTS, seed=CUSTOM_SEED)
    current_seed = students_df.attrs.get('seed', 'unknown')
    
    print("🏢 Generating consistent internship dataset...")
    internships_df = generate_consistent_internships()
    
    # Save datasets
    print("💾 Saving datasets...")
    students_df.to_csv('students_enhanced.csv', index=False)
    internships_df.to_csv('internships_enhanced.csv', index=False)
    
    # Save generation info
    gen_info = save_generation_info(current_seed, NUM_STUDENTS)
    
    # Display statistics
    print("\n📈 Dataset Statistics:")
    print(f"Students: {len(students_df)}")
    print(f"Internships: {len(internships_df)}")
    print(f"Total Capacity: {internships_df['capacity'].sum()}")
    print(f"Placement Potential: {min(100, internships_df['capacity'].sum() / len(students_df) * 100):.1f}%")
    
    print(f"\n🎯 Student Categories:")
    print(students_df['category'].value_counts())
    
    print(f"\n🏢 Company Tiers:")
    print(internships_df['company_tier'].value_counts())
    
    print(f"\n🎲 Generation Info:")
    print(f"Student Seed: {gen_info['student_seed']}")
    print(f"Internship Seed: {gen_info['internship_seed']} (fixed)")
    print(f"Generation Time: {gen_info['generation_time']}")
    
    print(f"\n✅ Enhanced datasets saved successfully!")
    print("📁 Files created:")
    print("  - students_enhanced.csv (unique each run)")
    print("  - internships_enhanced.csv (consistent)")
    print("  - generation_info.json (metadata)")

if __name__ == "__main__":
    main()