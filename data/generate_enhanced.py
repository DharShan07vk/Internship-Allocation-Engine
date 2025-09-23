"""
Enhanced Data Generator with Company Applications
==============================================

This script generates comprehensive data including:
- Students with company application preferences (UNIQUE each run)
- Internships with detailed information (CONSISTENT across runs)
- Real-time allocation tracking capabilities
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
import json
import os
from datetime import datetime, timedelta

# Dynamic seed based on current time for uniqueness
DYNAMIC_SEED = int(datetime.now().timestamp()) % 100000

# Fixed seed for consistent internships
INTERNSHIP_SEED = 42

def set_seeds(student_seed=None, internship_seed=INTERNSHIP_SEED):
    """Set seeds for data generation"""
    if student_seed is None:
        student_seed = DYNAMIC_SEED
    
    print(f"🎲 Using seeds - Students: {student_seed}, Internships: {internship_seed}")
    return student_seed, internship_seed

# Enhanced skill pools with more detailed categories
SKILLS_POOL = {
    'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby'],
    'web_development': ['react', 'angular', 'vue.js', 'node.js', 'django', 'flask', 'spring boot', 'express.js'],
    'mobile_development': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin', 'xamarin'],
    'data_science': ['machine learning', 'deep learning', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'r'],
    'cloud_devops': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
    'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle'],
    'ai_ml': ['natural language processing', 'computer vision', 'neural networks', 'reinforcement learning'],
    'business': ['business analysis', 'project management', 'excel', 'powerpoint', 'tableau', 'power bi', 'sap'],
    'cybersecurity': ['network security', 'penetration testing', 'cryptography', 'ethical hacking', 'firewall'],
    'design': ['ui/ux design', 'photoshop', 'figma', 'adobe illustrator', 'sketch', 'wireframing']
}

ALL_SKILLS = []
for category in SKILLS_POOL.values():
    ALL_SKILLS.extend(category)

# Enhanced company data with tiers and specializations
COMPANIES = {
    'Tier 1 - Tech Giants': {
        'companies': ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta', 'Netflix'],
        'specializations': ['ai_ml', 'programming', 'cloud_devops', 'data_science'],
        'avg_capacity': 8,
        'stipend_range': (80000, 150000)
    },
    'Tier 1 - Indian IT': {
        'companies': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'Tech Mahindra', 'Cognizant'],
        'specializations': ['programming', 'web_development', 'database', 'business'],
        'avg_capacity': 12,
        'stipend_range': (25000, 50000)
    },
    'Tier 2 - Startups': {
        'companies': ['Flipkart', 'Paytm', 'Zomato', 'Swiggy', 'BYJU\'S', 'Unacademy', 'PhonePe', 'Ola', 'Freshworks'],
        'specializations': ['web_development', 'mobile_development', 'data_science', 'ai_ml'],
        'avg_capacity': 6,
        'stipend_range': (35000, 80000)
    },
    'Tier 2 - Finance': {
        'companies': ['Goldman Sachs', 'JP Morgan', 'Morgan Stanley', 'HDFC Bank', 'ICICI Bank', 'Axis Bank'],
        'specializations': ['programming', 'data_science', 'business', 'cybersecurity'],
        'avg_capacity': 5,
        'stipend_range': (60000, 120000)
    },
    'Tier 3 - Consulting': {
        'companies': ['Deloitte', 'PwC', 'EY', 'KPMG', 'Accenture', 'McKinsey'],
        'specializations': ['business', 'data_science', 'programming', 'design'],
        'avg_capacity': 7,
        'stipend_range': (40000, 90000)
    }
}

ALL_COMPANIES = []
COMPANY_INFO = {}
for tier, data in COMPANIES.items():
    for company in data['companies']:
        ALL_COMPANIES.append(company)
        COMPANY_INFO[company] = {
            'tier': tier,
            'specializations': data['specializations'],
            'avg_capacity': data['avg_capacity'],
            'stipend_range': data['stipend_range']
        }

JOB_ROLES = {
    'programming': ['Software Engineer Intern', 'Backend Developer Intern', 'Full Stack Developer Intern'],
    'web_development': ['Frontend Developer Intern', 'Web Developer Intern', 'React Developer Intern'],
    'mobile_development': ['Mobile App Developer Intern', 'Android Developer Intern', 'iOS Developer Intern'],
    'data_science': ['Data Science Intern', 'ML Engineer Intern', 'Data Analyst Intern'],
    'ai_ml': ['AI Research Intern', 'Machine Learning Intern', 'Deep Learning Intern'],
    'cloud_devops': ['Cloud Engineer Intern', 'DevOps Engineer Intern', 'Site Reliability Intern'],
    'business': ['Business Analyst Intern', 'Product Management Intern', 'Strategy Intern'],
    'cybersecurity': ['Cybersecurity Intern', 'Security Analyst Intern', 'InfoSec Intern'],
    'design': ['UI/UX Designer Intern', 'Product Designer Intern', 'Graphic Designer Intern']
}

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat', 'Noida', 'Gurgaon']

EDUCATION = {
    'B.Tech CSE': 0.25,
    'B.Tech IT': 0.15, 
    'B.E Computer Science': 0.12,
    'B.Tech ECE': 0.08,
    'MCA': 0.10,
    'M.Tech CSE': 0.08,
    'M.Tech AI/ML': 0.05,
    'B.Sc Computer Science': 0.10,
    'MBA': 0.05,
    'BCA': 0.02
}

CATEGORIES = ['general', 'rural', 'reserved', 'female']

# Extended name pools for uniqueness
FIRST_NAMES = [
    'Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Sai', 'Reyansh', 'Ayaan', 'Krishna', 'Ishaan',
    'Shaurya', 'Atharv', 'Advaith', 'Arnav', 'Aahan', 'Aryan', 'Daksh', 'Darsh', 'Harsh', 'Dev',
    'Anaya', 'Diya', 'Priya', 'Ananya', 'Fatima', 'Anika', 'Kavya', 'Riya', 'Saanvi', 'Sara',
    'Myra', 'Aditi', 'Aadhya', 'Keya', 'Khushi', 'Pihu', 'Pari', 'Ira', 'Tara', 'Nisha',
    'Rohan', 'Kartik', 'Nikhil', 'Rahul', 'Amit', 'Suresh', 'Vikram', 'Ajay', 'Ravi', 'Karan',
    'Pooja', 'Sneha', 'Neha', 'Rekha', 'Sunita', 'Meena', 'Geeta', 'Sita', 'Rita', 'Lata'
]

LAST_NAMES = [
    'Sharma', 'Verma', 'Gupta', 'Singh', 'Kumar', 'Patel', 'Shah', 'Jain', 'Agarwal', 'Bansal',
    'Mehta', 'Malhotra', 'Kapoor', 'Chopra', 'Aggarwal', 'Goel', 'Jindal', 'Mittal', 'Goyal', 'Saxena',
    'Joshi', 'Tiwari', 'Mishra', 'Pandey', 'Yadav', 'Chauhan', 'Thakur', 'Rajput', 'Nair', 'Iyer'
]

COLLEGES = [
    'IIT Delhi', 'IIT Bombay', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur', 'IIT Roorkee', 'IIT Guwahati',
    'NIT Trichy', 'NIT Warangal', 'NIT Surathkal', 'NIT Calicut', 'BITS Pilani', 'IIIT Hyderabad',
    'VIT Vellore', 'SRM Chennai', 'Manipal Institute', 'PES University', 'RV College', 'BMS College'
]

def generate_unique_students(num_students=1000, seed=None):
    """Generate unique student dataset each time"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
    else:
        current_seed = DYNAMIC_SEED
        random.seed(current_seed)
        np.random.seed(current_seed)
        Faker.seed(current_seed)
        print(f"📊 Generating UNIQUE students with seed: {current_seed}")
    
    fake = Faker('en_IN')
    students = []
    
    for i in range(1, num_students + 1):
        # Generate unique name combinations
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        
        # Add middle initial sometimes for uniqueness
        if random.random() < 0.3:
            middle_initial = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            name = f"{first_name} {middle_initial}. {last_name}"
        else:
            name = f"{first_name} {last_name}"
        
        # Generate education based on weights
        education = np.random.choice(list(EDUCATION.keys()), p=list(EDUCATION.values()))
        
        # Generate skills based on education (more relevant skills)
        if 'M.Tech' in education or 'MCA' in education:
            num_skills = random.randint(5, 12)  # Advanced students have more skills
        elif 'MBA' in education:
            num_skills = random.randint(3, 8)
        else:
            num_skills = random.randint(4, 10)
        
        # Select skills with some domain focus
        primary_domain = random.choice(list(SKILLS_POOL.keys()))
        primary_skills = random.sample(SKILLS_POOL[primary_domain], min(num_skills//2, len(SKILLS_POOL[primary_domain])))
        remaining_skills = random.sample(ALL_SKILLS, num_skills - len(primary_skills))
        all_skills = list(set(primary_skills + remaining_skills))
        
        # Generate GPA with education bias
        if 'M.Tech' in education:
            gpa_base = random.uniform(7.5, 9.8)
        elif education in ['B.Tech CSE', 'B.E Computer Science']:
            gpa_base = random.uniform(6.5, 9.5)
        else:
            gpa_base = random.uniform(6.0, 9.0)
        
        gpa = round(gpa_base, 2)
        
        # Generate category with realistic distribution
        category = np.random.choice(CATEGORIES, p=[0.55, 0.20, 0.15, 0.10])
        
        # Generate company preferences (3-8 companies)
        num_preferences = random.randint(3, 8)
        preferred_companies = random.sample(ALL_COMPANIES, num_preferences)
        
        # Generate experience
        if 'M.Tech' in education or 'MCA' in education:
            past_internships = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        else:
            past_internships = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.20, 0.05])
        
        # Generate unique email variations
        email_variations = [
            f"{first_name.lower()}.{last_name.lower()}@student.edu",
            f"{first_name.lower()}{i}@college.ac.in",
            f"{last_name.lower()}.{first_name.lower()}@gmail.com",
            f"{first_name.lower()}{last_name.lower()}{i}@email.com"
        ]
        email = random.choice(email_variations)
        
        # Generate additional profile data
        github_profile = f"https://github.com/{first_name.lower()}{last_name.lower()}{i}" if random.random() > 0.3 else ""
        linkedin_profile = f"https://linkedin.com/in/{first_name.lower()}-{last_name.lower()}-{i}" if random.random() > 0.2 else ""
        
        student = {
            'id': f"STU{str(i).zfill(5)}",
            'name': name,
            'email': email,
            'phone': fake.phone_number(),
            'gpa': gpa,
            'skills': ', '.join(all_skills),
            'education': education,
            'college': random.choice(COLLEGES),
            'location': random.choice(CITIES),
            'category': category,
            'past_internships': past_internships,
            'preferred_companies': ', '.join(preferred_companies),
            'github_profile': github_profile,
            'linkedin_profile': linkedin_profile,
            'resume_score': round(random.uniform(60, 95), 1),
            'availability_start': (datetime.now() + timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d'),
            'preferred_duration': random.choice([3, 6, 12]),
            'expected_stipend': random.randint(15000, 80000),
            'willingness_to_relocate': random.choice(['Yes', 'No', 'Preferred']),
            'is_rural': category == 'rural',
            'is_female': category == 'female' or random.random() < 0.45,
            'experience': random.randint(0, 24),  # months
            'created_at': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        students.append(student)
    
    return pd.DataFrame(students)

def generate_consistent_internships(num_internships=200):
    """Generate consistent internship dataset (same every time)"""
    # Use fixed seed for consistent internships
    random.seed(INTERNSHIP_SEED)
    np.random.seed(INTERNSHIP_SEED)
    fake = Faker('en_IN')
    Faker.seed(INTERNSHIP_SEED)
    
    print(f"🏢 Generating CONSISTENT internships with seed: {INTERNSHIP_SEED}")
    
    internships = []
    for i in range(1, num_internships + 1):
        # Select company and get its info
        company = random.choice(ALL_COMPANIES)
        company_data = COMPANY_INFO[company]
        
        # Select specialization based on company focus
        specialization = random.choice(company_data['specializations'])
        
        # Generate role based on specialization
        if specialization in JOB_ROLES:
            role = random.choice(JOB_ROLES[specialization])
        else:
            role = random.choice(JOB_ROLES['programming'])
        
        # Generate required skills based on specialization
        if specialization in SKILLS_POOL:
            core_skills = random.sample(SKILLS_POOL[specialization], min(3, len(SKILLS_POOL[specialization])))
            additional_skills = random.sample(ALL_SKILLS, random.randint(2, 5))
            required_skills = list(set(core_skills + additional_skills))
        else:
            required_skills = random.sample(ALL_SKILLS, random.randint(3, 8))
        
        # Generate capacity based on company tier
        base_capacity = company_data['avg_capacity']
        capacity = max(1, base_capacity + random.randint(-3, 4))
        
        # Generate stipend based on company tier
        stipend_min, stipend_max = company_data['stipend_range']
        stipend = random.randint(stipend_min, stipend_max)
        
        # Generate other attributes
        duration_months = random.choice([3, 6, 12])
        
        # Reserved quota
        reserved_for = np.random.choice(
            ['none', 'rural', 'reserved', 'female'], 
            p=[0.7, 0.1, 0.1, 0.1]
        )
        
        # Application deadline
        deadline = (datetime.now() + timedelta(days=random.randint(15, 60))).strftime('%Y-%m-%d')
        
        # Start date
        start_date = (datetime.now() + timedelta(days=random.randint(60, 120))).strftime('%Y-%m-%d')
        
        internship = {
            'id': f"INT{str(i).zfill(5)}",
            'company': company,
            'company_tier': company_data['tier'],
            'title': role,
            'department': specialization.replace('_', ' ').title(),
            'skills_required': ', '.join(required_skills),
            'capacity': capacity,
            'location': random.choice(CITIES),
            'work_mode': np.random.choice(['On-site', 'Remote', 'Hybrid'], p=[0.6, 0.2, 0.2]),
            'reserved_for': reserved_for,
            'duration_months': duration_months,
            'stipend': stipend,
            'description': f"Exciting {role} opportunity at {company} working on {specialization} projects.",
            'requirements': f"Strong skills in {', '.join(required_skills[:3])} required.",
            'application_deadline': deadline,
            'internship_start_date': start_date,
            'min_gpa': round(random.uniform(6.0, 8.0), 1),
            'preferred_year': random.choice(['3rd Year', '4th Year', 'Final Year', 'Any']),
            'application_count': 0,
            'is_active': True,
            'created_at': (datetime.now() - timedelta(days=random.randint(5, 45))).isoformat()
        }
        internships.append(internship)
    
    return pd.DataFrame(internships)

def update_application_counts(students_df, internships_df):
    """Update application counts based on student preferences"""
    print("🔄 Updating application counts...")
    
    application_counts = {}
    
    for _, student in students_df.iterrows():
        if pd.notna(student['preferred_companies']):
            preferred = [c.strip() for c in str(student['preferred_companies']).split(',')]
            for company in preferred:
                if company in application_counts:
                    application_counts[company] += 1
                else:
                    application_counts[company] = 1
    
    # Update internships dataframe
    for idx, internship in internships_df.iterrows():
        company = internship['company']
        count = application_counts.get(company, 0)
        internships_df.at[idx, 'application_count'] = count
    
    return internships_df

def save_generation_metadata(student_seed, internship_seed, num_students, num_internships):
    """Save generation metadata"""
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'student_seed': student_seed,
        'internship_seed': internship_seed,
        'num_students': num_students,
        'num_internships': num_internships,
        'total_capacity': 0,  # Will be updated
        'placement_potential': 0,  # Will be updated
        'version': '2.0_enhanced',
        'uniqueness': 'students_unique_each_run',
        'consistency': 'internships_same_each_run'
    }
    
    with open('generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    """Generate comprehensive enhanced data with uniqueness"""
    print("🚀 Enhanced Data Generator - Unique Students + Consistent Internships")
    print("=" * 70)
    
    # Configuration
    NUM_STUDENTS = 1000
    NUM_INTERNSHIPS = 200
    CUSTOM_STUDENT_SEED = None  # Set to number for reproducible students
    
    # Set seeds
    student_seed, internship_seed = set_seeds(CUSTOM_STUDENT_SEED, INTERNSHIP_SEED)
    
    # Generate datasets
    print(f"\n📊 Generating {NUM_STUDENTS} unique students...")
    students_df = generate_unique_students(NUM_STUDENTS, seed=student_seed)
    
    print(f"\n🏢 Generating {NUM_INTERNSHIPS} consistent internships...")
    internships_df = generate_consistent_internships(NUM_INTERNSHIPS)
    
    # Update application counts
    internships_df = update_application_counts(students_df, internships_df)
    
    # Calculate statistics
    total_capacity = internships_df['capacity'].sum()
    placement_potential = min(100, total_capacity / len(students_df) * 100)
    
    # Save datasets in current directory (data folder)
    print(f"\n💾 Saving datasets in current directory...")
    students_df.to_csv('students_enhanced.csv', index=False)
    internships_df.to_csv('internships_enhanced.csv', index=False)
    
    # Save metadata
    metadata = save_generation_metadata(student_seed, internship_seed, NUM_STUDENTS, NUM_INTERNSHIPS)
    metadata['total_capacity'] = int(total_capacity)
    metadata['placement_potential'] = round(placement_potential, 1)
    
    with open('generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate tracking data
    tracking_data = {
        'total_applications': 0,
        'processing_applications': 0,
        'completed_allocations': 0,
        'pending_reviews': 0,
        'last_updated': datetime.now().isoformat(),
        'allocation_status': 'pending',
        'allocation_round': 1,
        'generation_info': {
            'student_seed': student_seed,
            'generation_time': datetime.now().isoformat(),
            'is_unique': True
        }
    }
    
    with open('allocation_tracking.json', 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    # Print comprehensive summary
    print(f"\n✅ Enhanced Data Generated Successfully!")
    print(f"📁 Files created in current directory:")
    print(f"   - students_enhanced.csv ({len(students_df)} students)")
    print(f"   - internships_enhanced.csv ({len(internships_df)} internships)")
    print(f"   - generation_metadata.json (generation info)")
    print(f"   - allocation_tracking.json (tracking data)")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Students: {len(students_df)} (UNIQUE each run)")
    print(f"   Total Internships: {len(internships_df)} (CONSISTENT each run)")
    print(f"   Total Capacity: {total_capacity}")
    print(f"   Placement Potential: {placement_potential:.1f}%")
    
    print(f"\n🎯 Student Categories:")
    category_counts = students_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category.title()}: {count} ({count/len(students_df)*100:.1f}%)")
    
    print(f"\n🏢 Company Tiers:")
    tier_counts = internships_df['company_tier'].value_counts()
    for tier, count in tier_counts.items():
        print(f"   {tier}: {count} internships")
    
    print(f"\n🔄 Application Statistics:")
    total_applications = internships_df['application_count'].sum()
    avg_applications = internships_df['application_count'].mean()
    print(f"   Total Applications: {total_applications}")
    print(f"   Average per Internship: {avg_applications:.1f}")
    
    print(f"\n🎲 Generation Seeds:")
    print(f"   Student Seed: {student_seed} (changes each run)")
    print(f"   Internship Seed: {internship_seed} (fixed for consistency)")
    
    print(f"\n🚀 Ready for allocation engine!")

if __name__ == "__main__":
    main()
