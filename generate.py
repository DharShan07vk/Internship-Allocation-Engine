"""
Enhanced Data Generation for Internship Allocation Engine
========================================================

Generates realistic, comprehensive datasets for students and internships
with sophisticated skill matching, diversity representation, and real-world
distribution patterns.

Features:
- Realistic Indian names, cities, and educational institutions
- Sophisticated skill combinations based on career tracks
- Proper category distribution reflecting real demographics
- Company and internship data based on actual industry patterns
- Configurable scale for testing different dataset sizes
- Data quality validation and statistics

Usage:
    python data/generate.py --students 1000 --internships 200
    python data/generate.py --scale large --validate
"""

import pandas as pd
import numpy as np
import random
import argparse
import os
from typing import List, Dict, Tuple
from faker import Faker

# Configure faker for Indian context
fake = Faker(['en_IN', 'hi_IN'])
Faker.seed(42)  # For reproducible data
random.seed(42)
np.random.seed(42)

class DataGenerator:
    """Advanced data generator for internship allocation system"""
    
    def __init__(self):
        self.setup_data_templates()
    
    def setup_data_templates(self):
        """Setup comprehensive data templates"""
        
        # Skill categories and progressions
        self.skill_categories = {
            'software_engineering': {
                'core': ['python', 'java', 'javascript', 'git', 'sql', 'data structures'],
                'web': ['html', 'css', 'react', 'node.js', 'express', 'mongodb', 'rest api'],
                'backend': ['spring boot', 'django', 'flask', 'microservices', 'docker', 'kubernetes'],
                'mobile': ['android', 'kotlin', 'swift', 'react native', 'flutter'],
                'advanced': ['system design', 'cloud computing', 'devops', 'testing']
            },
            'data_science': {
                'core': ['python', 'r', 'sql', 'statistics', 'mathematics'],
                'ml': ['machine learning', 'scikit-learn', 'pandas', 'numpy', 'matplotlib'],
                'advanced': ['deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision'],
                'tools': ['jupyter', 'tableau', 'power bi', 'hadoop', 'spark', 'aws'],
                'domain': ['data visualization', 'feature engineering', 'model deployment']
            },
            'business_analyst': {
                'core': ['excel', 'powerpoint', 'business analysis', 'requirements gathering'],
                'data': ['sql', 'tableau', 'power bi', 'data analysis', 'statistics'],
                'process': ['process improvement', 'project management', 'agile', 'scrum'],
                'domain': ['market research', 'competitive analysis', 'stakeholder management']
            },
            'digital_marketing': {
                'core': ['digital marketing', 'social media marketing', 'content marketing'],
                'analytics': ['google analytics', 'seo', 'sem', 'conversion optimization'],
                'tools': ['facebook ads', 'google ads', 'email marketing', 'marketing automation'],
                'creative': ['content creation', 'copywriting', 'graphic design', 'video editing']
            },
            'cybersecurity': {
                'core': ['network security', 'information security', 'cybersecurity fundamentals'],
                'technical': ['penetration testing', 'vulnerability assessment', 'incident response'],
                'tools': ['wireshark', 'metasploit', 'nessus', 'burp suite', 'kali linux'],
                'compliance': ['iso 27001', 'gdpr', 'compliance', 'risk assessment']
            },
            'finance': {
                'core': ['financial analysis', 'accounting', 'excel', 'financial modeling'],
                'investment': ['equity research', 'portfolio management', 'risk management'],
                'corporate': ['corporate finance', 'valuation', 'mergers acquisitions', 'budgeting'],
                'fintech': ['python', 'sql', 'financial technology', 'algorithmic trading']
            }
        }
        
        # Indian educational institutions
        self.educational_institutions = {
            'engineering': ['IIT Delhi', 'IIT Bombay', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                          'NIT Trichy', 'NIT Warangal', 'BITS Pilani', 'VIT Vellore', 'SRM University',
                          'Manipal Institute', 'PES University', 'RV College of Engineering',
                          'Delhi Technological University', 'Jadavpur University'],
            'management': ['IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                         'ISB Hyderabad', 'XLRI Jamshedpur', 'FMS Delhi', 'JBIMS Mumbai',
                         'IIM Indore', 'MDI Gurgaon'],
            'commerce': ['Delhi University', 'Mumbai University', 'Calcutta University',
                       'Christ University', 'Loyola College Chennai', 'Madras Christian College',
                       'Hansraj College', 'SRCC Delhi', 'St. Xaviers Mumbai'],
            'science': ['Indian Institute of Science', 'Delhi University', 'Jawaharlal Nehru University',
                      'Banaras Hindu University', 'University of Hyderabad', 'Pune University']
        }
        
        # Company categories with realistic distribution
        self.companies = {
            'tech_giants': ['Google', 'Microsoft', 'Amazon', 'Facebook', 'Apple', 'Netflix'],
            'indian_it': ['TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'Tech Mahindra', 'Mindtree',
                        'Cognizant', 'Capgemini', 'L&T Infotech', 'Mphasis'],
            'startups': ['Flipkart', 'Paytm', 'Zomato', 'Swiggy', 'Ola', 'BYJU\'S', 'Unacademy',
                       'PhonePe', 'PolicyBazaar', 'Freshworks', 'Razorpay', 'CRED'],
            'consulting': ['McKinsey & Company', 'Boston Consulting Group', 'Bain & Company',
                         'Deloitte', 'PwC', 'EY', 'KPMG', 'Accenture'],
            'finance': ['Goldman Sachs', 'Morgan Stanley', 'JP Morgan', 'Citi', 'HDFC Bank',
                      'ICICI Bank', 'Axis Bank', 'Kotak Mahindra Bank', 'Zerodha'],
            'product': ['Atlassian', 'Slack', 'Zoom', 'Adobe', 'Salesforce', 'ServiceNow',
                      'Uber', 'Airbnb', 'Spotify', 'LinkedIn']
        }
        
        # Indian cities with tier classification
        self.cities = {
            'tier1': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune'],
            'tier2': ['Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore',
                    'Bhopal', 'Visakhapatnam', 'Coimbatore', 'Kochi', 'Thiruvananthapuram'],
            'tier3': ['Dehradun', 'Mysore', 'Meerut', 'Guwahati', 'Chandigarh', 'Nashik',
                    'Vadodara', 'Agra', 'Madurai', 'Ludhiana', 'Jalandhar', 'Amritsar']
        }
        
        # Degree programs
        self.degree_programs = {
            'engineering': ['B.Tech CSE', 'B.Tech IT', 'B.Tech ECE', 'B.Tech Mechanical',
                          'B.Tech Civil', 'B.E Computer Science', 'B.E Electronics',
                          'M.Tech CSE', 'M.Tech Data Science', 'M.Tech AI'],
            'management': ['MBA', 'BBA', 'B.Com', 'M.Com', 'PGDM'],
            'science': ['B.Sc Computer Science', 'B.Sc Mathematics', 'B.Sc Physics',
                      'M.Sc Data Science', 'M.Sc Statistics', 'MCA'],
            'arts': ['BA Economics', 'BA Psychology', 'BA English', 'MA Economics']
        }
    
    def generate_student_profile(self, student_id: int) -> Dict:
        """Generate a comprehensive student profile"""
        
        # Basic demographics
        name = fake.name()
        email = f"{name.lower().replace(' ', '.')}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}"
        
        # Academic performance with realistic distribution
        # Higher GPA students more likely to have better opportunities
        gpa = np.random.beta(7, 3) * 4 + 6  # Beta distribution shifted to 6-10 range
        gpa = min(10.0, max(6.0, round(gpa, 2)))
        
        # Education background influences skill set
        education_type = np.random.choice(['engineering', 'management', 'science', 'arts'], 
                                        p=[0.4, 0.25, 0.25, 0.1])
        
        institution_type = education_type if education_type in self.educational_institutions else 'engineering'
        education = random.choice(self.degree_programs[education_type])
        
        # Location with realistic urban-rural distribution
        location_type = np.random.choice(['tier1', 'tier2', 'tier3'], p=[0.4, 0.35, 0.25])
        location = random.choice(self.cities[location_type])
        
        # Category with realistic demographic distribution
        category = np.random.choice(['general', 'rural', 'reserved', 'female'], 
                                  p=[0.45, 0.20, 0.20, 0.15])
        
        # Past experience based on academic performance and category
        experience_prob = min(0.8, (gpa - 6) / 4 * 0.6 + 0.1)
        if category in ['rural', 'reserved']:
            experience_prob *= 0.7  # Slightly lower experience for underrepresented groups
        
        past_internships = np.random.binomial(3, experience_prob)
        
        # Skills based on education and performance
        skills = self._generate_skills_for_student(education_type, gpa, past_internships)
        
        return {
            'id': student_id,
            'name': name,
            'email': email,
            'gpa': gpa,
            'skills': ', '.join(skills),
            'education': education,
            'location': location,
            'category': category,
            'past_internships': past_internships
        }
    
    def _generate_skills_for_student(self, education_type: str, gpa: float, experience: int) -> List[str]:
        """Generate realistic skill combinations for a student"""
        
        skills = set()
        
        # Map education to skill category
        skill_category_map = {
            'engineering': ['software_engineering', 'data_science'],
            'management': ['business_analyst', 'digital_marketing'],
            'science': ['data_science', 'software_engineering'],
            'arts': ['digital_marketing', 'business_analyst']
        }
        
        primary_categories = skill_category_map.get(education_type, ['software_engineering'])
        primary_category = random.choice(primary_categories)
        
        # Skill acquisition based on GPA and experience
        skill_breadth = int((gpa - 6) / 4 * 6) + experience + random.randint(3, 8)
        
        # Add core skills first
        core_skills = self.skill_categories[primary_category]['core']
        skills.update(random.sample(core_skills, min(len(core_skills), 4)))
        
        # Add specialized skills based on performance
        for skill_level, skill_list in self.skill_categories[primary_category].items():
            if skill_level != 'core':
                num_skills = min(len(skill_list), max(1, int(skill_breadth / len(self.skill_categories[primary_category]))))
                if gpa >= 8.0 or experience >= 2:  # High performers get more advanced skills
                    num_skills += 1
                skills.update(random.sample(skill_list, num_skills))
        
        # Cross-domain skills for versatile candidates
        if len(skills) < skill_breadth and random.random() < 0.3:
            other_categories = [cat for cat in self.skill_categories.keys() if cat != primary_category]
            other_category = random.choice(other_categories)
            other_skills = self.skill_categories[other_category]['core']
            skills.update(random.sample(other_skills, min(2, len(other_skills))))
        
        return list(skills)[:skill_breadth]  # Limit total skills
    
    def generate_internship_profile(self, internship_id: int) -> Dict:
        """Generate a comprehensive internship profile"""
        
        # Company selection with realistic distribution
        company_type = np.random.choice(list(self.companies.keys()), 
                                      p=[0.1, 0.35, 0.25, 0.1, 0.1, 0.1])
        company = random.choice(self.companies[company_type])
        
        # Job titles based on company type
        title_map = {
            'tech_giants': ['Software Engineer Intern', 'Data Science Intern', 'Product Management Intern',
                          'SWE Intern', 'ML Engineering Intern', 'Full Stack Developer Intern'],
            'indian_it': ['Software Developer Intern', 'Associate Software Engineer', 'Technology Analyst',
                        'Systems Engineer Intern', 'Full Stack Developer Intern'],
            'startups': ['Software Engineering Intern', 'Product Intern', 'Growth Intern',
                       'Data Science Intern', 'Backend Developer Intern', 'Frontend Developer Intern'],
            'consulting': ['Business Analyst Intern', 'Consultant Intern', 'Strategy Intern',
                         'Management Consultant Intern', 'Research Analyst Intern'],
            'finance': ['Investment Banking Analyst', 'Equity Research Intern', 'Risk Analyst Intern',
                      'Financial Analyst Intern', 'Quantitative Analyst Intern'],
            'product': ['Product Management Intern', 'Product Design Intern', 'Product Marketing Intern',
                      'UX Design Intern', 'Product Analyst Intern']
        }
        
        title = random.choice(title_map[company_type])
        
        # Skills required based on job title and company
        required_skills = self._generate_skills_for_internship(title, company_type)
        
        # Capacity based on company size and type
        capacity_map = {
            'tech_giants': (8, 15),
            'indian_it': (10, 20),
            'startups': (2, 8),
            'consulting': (3, 10),
            'finance': (2, 6),
            'product': (3, 8)
        }
        
        min_cap, max_cap = capacity_map[company_type]
        capacity = random.randint(min_cap, max_cap)
        
        # Location preference (companies more likely to be in tier 1 cities)
        if company_type in ['tech_giants', 'consulting', 'finance']:
            location = random.choice(self.cities['tier1'])
        else:
            location_type = np.random.choice(['tier1', 'tier2'], p=[0.7, 0.3])
            location = random.choice(self.cities[location_type])
        
        # Reserved positions (some internships reserved for diversity)
        reserved_for = np.random.choice(['none', 'rural', 'reserved', 'female'], 
                                      p=[0.7, 0.1, 0.1, 0.1])
        
        # Additional attributes
        duration = np.random.choice([2, 3, 6], p=[0.3, 0.5, 0.2])  # months
        
        stipend_ranges = {
            'tech_giants': (50000, 100000),
            'indian_it': (15000, 35000),
            'startups': (10000, 50000),
            'consulting': (40000, 80000),
            'finance': (30000, 70000),
            'product': (25000, 60000)
        }
        
        min_stipend, max_stipend = stipend_ranges[company_type]
        stipend = random.randint(min_stipend, max_stipend)
        
        # Description
        descriptions = [
            f"Join our {title.lower()} program and work on cutting-edge projects",
            f"Exciting opportunity to work with latest technologies in {title.lower()} role",
            f"Hands-on experience in {company}'s innovative {title.lower()} team",
            f"Learn from industry experts in our {title.lower()} internship program"
        ]
        
        return {
            'id': internship_id,
            'company': company,
            'title': title,
            'skills_required': ', '.join(required_skills),
            'capacity': capacity,
            'location': location,
            'reserved_for': reserved_for,
            'description': random.choice(descriptions),
            'duration_months': duration,
            'stipend': stipend,
            'is_active': True
        }
    
    def _generate_skills_for_internship(self, title: str, company_type: str) -> List[str]:
        """Generate required skills for an internship based on title and company"""
        
        skills = set()
        
        # Map job titles to skill requirements
        if 'software' in title.lower() or 'swe' in title.lower() or 'developer' in title.lower():
            primary_category = 'software_engineering'
            if 'full stack' in title.lower():
                skills.update(random.sample(self.skill_categories[primary_category]['web'], 3))
                skills.update(random.sample(self.skill_categories[primary_category]['backend'], 2))
            elif 'backend' in title.lower():
                skills.update(random.sample(self.skill_categories[primary_category]['backend'], 4))
            elif 'frontend' in title.lower():
                skills.update(random.sample(self.skill_categories[primary_category]['web'], 4))
            else:
                skills.update(random.sample(self.skill_categories[primary_category]['core'], 4))
                
        elif 'data' in title.lower() or 'ml' in title.lower():
            primary_category = 'data_science'
            skills.update(random.sample(self.skill_categories[primary_category]['core'], 3))
            skills.update(random.sample(self.skill_categories[primary_category]['ml'], 3))
            if 'ml' in title.lower():
                skills.update(random.sample(self.skill_categories[primary_category]['advanced'], 2))
                
        elif 'product' in title.lower():
            skills.update(['product management', 'analytics', 'user research', 'agile'])
            skills.update(random.sample(self.skill_categories['business_analyst']['core'], 2))
            
        elif 'business' in title.lower() or 'analyst' in title.lower():
            primary_category = 'business_analyst'
            skills.update(random.sample(self.skill_categories[primary_category]['core'], 4))
            skills.update(random.sample(self.skill_categories[primary_category]['data'], 2))
            
        elif 'marketing' in title.lower() or 'growth' in title.lower():
            primary_category = 'digital_marketing'
            skills.update(random.sample(self.skill_categories[primary_category]['core'], 3))
            skills.update(random.sample(self.skill_categories[primary_category]['analytics'], 2))
            
        elif 'finance' in title.lower() or 'investment' in title.lower():
            primary_category = 'finance'
            skills.update(random.sample(self.skill_categories[primary_category]['core'], 3))
            if 'investment' in title.lower():
                skills.update(random.sample(self.skill_categories[primary_category]['investment'], 2))
                
        else:
            # Default to software engineering
            primary_category = 'software_engineering'
            skills.update(random.sample(self.skill_categories[primary_category]['core'], 4))
        
        # Add company-specific requirements
        if company_type == 'tech_giants':
            skills.update(['problem solving', 'system design'])
        elif company_type == 'startups':
            skills.update(['agile', 'fast-paced environment'])
        elif company_type == 'consulting':
            skills.update(['presentation skills', 'client communication'])
        
        # Ensure minimum skill requirements
        if len(skills) < 3:
            core_skills = self.skill_categories['software_engineering']['core']
            skills.update(random.sample(core_skills, 3 - len(skills)))
        
        return list(skills)
    
    def generate_datasets(self, num_students: int = 500, num_internships: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete datasets"""
        
        print(f"Generating {num_students} student profiles...")
        students_data = []
        for i in range(1, num_students + 1):
            if i % 100 == 0:
                print(f"  Generated {i} students")
            students_data.append(self.generate_student_profile(i))
        
        print(f"Generating {num_internships} internship profiles...")
        internships_data = []
        for i in range(1, num_internships + 1):
            if i % 20 == 0:
                print(f"  Generated {i} internships")
            internships_data.append(self.generate_internship_profile(i))
        
        students_df = pd.DataFrame(students_data)
        internships_df = pd.DataFrame(internships_data)
        
        return students_df, internships_df
    
    def validate_data_quality(self, students_df: pd.DataFrame, internships_df: pd.DataFrame) -> Dict:
        """Validate and analyze data quality"""
        
        print("\n📊 Data Quality Analysis:")
        
        # Students analysis
        student_stats = {
            'total_students': len(students_df),
            'avg_gpa': round(students_df['gpa'].mean(), 2),
            'category_distribution': students_df['category'].value_counts().to_dict(),
            'location_distribution': students_df['location'].value_counts().to_dict(),
            'avg_skills_per_student': round(students_df['skills'].str.count(',').mean() + 1, 1),
            'experience_distribution': students_df['past_internships'].value_counts().to_dict()
        }
        
        # Internships analysis
        internship_stats = {
            'total_internships': len(internships_df),
            'total_capacity': int(internships_df['capacity'].sum()),
            'avg_capacity_per_internship': round(internships_df['capacity'].mean(), 1),
            'company_distribution': internships_df['company'].value_counts().to_dict(),
            'title_distribution': internships_df['title'].value_counts().to_dict(),
            'avg_skills_per_internship': round(internships_df['skills_required'].str.count(',').mean() + 1, 1),
            'avg_stipend': round(internships_df['stipend'].mean()),
            'reserved_distribution': internships_df['reserved_for'].value_counts().to_dict()
        }
        
        # Supply-demand analysis
        supply_demand = {
            'student_internship_ratio': round(len(students_df) / len(internships_df), 2),
            'student_capacity_ratio': round(len(students_df) / internships_df['capacity'].sum(), 2),
            'theoretical_placement_rate': min(1.0, internships_df['capacity'].sum() / len(students_df))
        }
        
        # Print summary
        print(f"  👥 Students: {student_stats['total_students']}")
        print(f"  🏢 Internships: {internship_stats['total_internships']}")
        print(f"  🎯 Total Capacity: {internship_stats['total_capacity']}")
        print(f"  📈 Theoretical Placement Rate: {supply_demand['theoretical_placement_rate']:.2%}")
        print(f"  📊 Avg GPA: {student_stats['avg_gpa']}")
        print(f"  💰 Avg Stipend: ₹{internship_stats['avg_stipend']:,}")
        
        print(f"\n  Category Distribution:")
        for category, count in student_stats['category_distribution'].items():
            percentage = count / student_stats['total_students'] * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")
        
        return {
            'student_stats': student_stats,
            'internship_stats': internship_stats,
            'supply_demand': supply_demand
        }
    
    def save_datasets(self, students_df: pd.DataFrame, internships_df: pd.DataFrame, 
                     output_dir: str = "data") -> List[str]:
        """Save datasets with validation"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main datasets
        students_path = os.path.join(output_dir, "students.csv")
        internships_path = os.path.join(output_dir, "internships.csv")
        
        students_df.to_csv(students_path, index=False)
        internships_df.to_csv(internships_path, index=False)
        
        # Save sample datasets for quick testing
        students_sample = students_df.sample(min(100, len(students_df)))
        internships_sample = internships_df.sample(min(20, len(internships_df)))
        
        students_sample.to_csv(os.path.join(output_dir, "students_sample.csv"), index=False)
        internships_sample.to_csv(os.path.join(output_dir, "internships_sample.csv"), index=False)
        
        saved_files = [students_path, internships_path]
        
        return saved_files

def main():
    """Main function with CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive datasets for Internship Allocation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default datasets
  python data/generate.py
  
  # Generate large scale datasets
  python data/generate.py --students 5000 --internships 1000
  
  # Generate and validate with analysis
  python data/generate.py --students 1000 --internships 200 --validate
  
  # Quick test datasets
  python data/generate.py --scale small
        """
    )
    
    # Dataset size options
    parser.add_argument("--students", type=int, default=500, 
                       help="Number of students to generate (default: 500)")
    parser.add_argument("--internships", type=int, default=100,
                       help="Number of internships to generate (default: 100)")
    
    # Preset scales
    parser.add_argument("--scale", choices=['small', 'medium', 'large', 'xlarge'],
                       help="Preset dataset scales (overrides individual counts)")
    
    # Options
    parser.add_argument("--validate", action="store_true",
                       help="Run comprehensive data validation and analysis")
    parser.add_argument("--output-dir", default="data",
                       help="Output directory for generated files (default: data)")
    
    args = parser.parse_args()
    
    # Apply preset scales
    scale_configs = {
        'small': (100, 20),
        'medium': (500, 100),
        'large': (2000, 400),
        'xlarge': (10000, 2000)
    }
    
    if args.scale:
        args.students, args.internships = scale_configs[args.scale]
        print(f"Using {args.scale} scale: {args.students} students, {args.internships} internships")
    
    # Validate inputs
    if args.students <= 0 or args.internships <= 0:
        print("Error: Student and internship counts must be positive")
        return 1
    
    # Generate datasets
    print("🚀 Starting Enhanced Data Generation...")
    print("=" * 60)
    
    generator = DataGenerator()
    
    try:
        students_df, internships_df = generator.generate_datasets(args.students, args.internships)
        
        # Validation and analysis
        if args.validate:
            quality_report = generator.validate_data_quality(students_df, internships_df)
        
        # Save datasets
        print(f"\n💾 Saving datasets to '{args.output_dir}'...")
        saved_files = generator.save_datasets(students_df, internships_df, args.output_dir)
        
        print("\n✅ Data generation completed successfully!")
        print(f"📁 Generated files:")
        for file_path in saved_files:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file_path} ({file_size:.1f} KB)")
        
        print(f"\n🎯 Ready for allocation pipeline!")
        print(f"   Run: python runPython.py --train-model --force-embeddings")
        
        return 0
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
