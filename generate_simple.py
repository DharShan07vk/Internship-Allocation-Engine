"""Simple data generator for quick testing"""

import pandas as pd
import numpy as np
import random
from faker import Faker

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
Faker.seed(42)

fake = Faker('en_IN')

# Define skill pools
SKILLS_POOL = {
    'core_programming': ['python', 'java', 'javascript', 'c++', 'sql', 'git', 'html', 'css'],
    'web_development': ['react', 'node.js', 'django', 'flask', 'mongodb', 'postgresql', 'rest api'],
    'data_science': ['machine learning', 'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'tensorflow', 'pytorch'],
    'business': ['excel', 'powerpoint', 'business analysis', 'project management', 'tableau', 'power bi'],
    'cloud_devops': ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform'],
    'mobile': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin']
}

ALL_SKILLS = []
for category in SKILLS_POOL.values():
    ALL_SKILLS.extend(category)

COMPANIES = [
    'TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'Tech Mahindra',
    'Google', 'Microsoft', 'Amazon', 'Flipkart', 'Paytm',
    'Zomato', 'Swiggy', 'BYJU\'S', 'Unacademy', 'PhonePe',
    'Deloitte', 'PwC', 'EY', 'KPMG', 'Accenture',
    'Goldman Sachs', 'JP Morgan', 'HDFC Bank', 'ICICI Bank'
]

JOB_TITLES = [
    'Software Engineer Intern', 'Data Science Intern', 'Web Developer Intern',
    'Business Analyst Intern', 'Product Management Intern', 'Cloud Engineer Intern',
    'Full Stack Developer Intern', 'Backend Developer Intern', 'Frontend Developer Intern',
    'Mobile Developer Intern', 'DevOps Engineer Intern', 'Machine Learning Intern'
]

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat']

EDUCATION = ['B.Tech CSE', 'B.Tech IT', 'B.E Computer Science', 'MCA', 'B.Sc Computer Science', 'M.Tech AI', 'MBA']

CATEGORIES = ['general', 'rural', 'reserved', 'female']

def generate_students(num_students=500):
    """Generate student data"""
    print(f"Generating {num_students} students...")
    
    students = []
    for i in range(1, num_students + 1):
        # Select skills (3-8 skills per student)
        num_skills = random.randint(3, 8)
        skills = random.sample(ALL_SKILLS, num_skills)
        
        student = {
            'id': i,
            'name': fake.name(),
            'email': f"student{i}@example.com",
            'gpa': round(random.uniform(6.0, 9.8), 2),
            'skills': ', '.join(skills),
            'education': random.choice(EDUCATION),
            'location': random.choice(CITIES),
            'category': np.random.choice(CATEGORIES, p=[0.5, 0.2, 0.2, 0.1]),
            'past_internships': np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        }
        students.append(student)
    
    return pd.DataFrame(students)

def generate_internships(num_internships=100):
    """Generate internship data"""
    print(f"Generating {num_internships} internships...")
    
    internships = []
    for i in range(1, num_internships + 1):
        # Select required skills (3-6 skills per internship)
        num_skills = random.randint(3, 6)
        required_skills = random.sample(ALL_SKILLS, num_skills)
        
        internship = {
            'id': i,
            'company': random.choice(COMPANIES),
            'title': random.choice(JOB_TITLES),
            'skills_required': ', '.join(required_skills),
            'capacity': random.randint(2, 8),
            'location': random.choice(CITIES),
            'reserved_for': np.random.choice(['none', 'rural', 'reserved', 'female'], p=[0.7, 0.1, 0.1, 0.1])
        }
        internships.append(internship)
    
    return pd.DataFrame(internships)

def main():
    """Generate and save data"""
    print("🚀 Generating Sample Data...")
    
    # Generate data
    students_df = generate_students(500)
    internships_df = generate_internships(100)
    
    # Save to CSV
    students_df.to_csv('data/students.csv', index=False)
    internships_df.to_csv('data/internships.csv', index=False)
    
    # Print summary
    print(f"\n✅ Generated Data:")
    print(f"   Students: {len(students_df)}")
    print(f"   Internships: {len(internships_df)}")
    print(f"   Total Capacity: {internships_df['capacity'].sum()}")
    print(f"   Theoretical Placement Rate: {internships_df['capacity'].sum() / len(students_df):.2%}")
    
    print(f"\n📊 Category Distribution:")
    category_counts = students_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} ({count/len(students_df)*100:.1f}%)")
    
    print(f"\n✅ Files saved:")
    print(f"   - data/students.csv")
    print(f"   - data/internships.csv")

if __name__ == "__main__":
    main()
