"""Quick dataset generation script with different presets"""

import os
import sys
from generate_enhanced import generate_unique_students, generate_consistent_internships, update_application_counts, save_generation_metadata
import json
from datetime import datetime

def generate_preset(preset_name):
    """Generate datasets with predefined configurations"""
    
    presets = {
        'small': {'students': 100, 'internships': 50},
        'medium': {'students': 500, 'internships': 100}, 
        'large': {'students': 1000, 'internships': 200},
        'xlarge': {'students': 2000, 'internships': 400},
        'test': {'students': 50, 'internships': 25}
    }
    
    if preset_name not in presets:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return
    
    config = presets[preset_name]
    print(f"🎯 Generating {preset_name.upper()} dataset...")
    print(f"   Students: {config['students']}")
    print(f"   Internships: {config['internships']}")
    
    # Generate data
    students_df = generate_unique_students(config['students'])
    internships_df = generate_consistent_internships(config['internships'])
    internships_df = update_application_counts(students_df, internships_df)
    
    # Save with preset name
    students_file = f'students_{preset_name}.csv'
    internships_file = f'internships_{preset_name}.csv'
    
    students_df.to_csv(students_file, index=False)
    internships_df.to_csv(internships_file, index=False)
    
    # Also save as enhanced (for compatibility)
    students_df.to_csv('students_enhanced.csv', index=False)
    internships_df.to_csv('internships_enhanced.csv', index=False)
    
    print(f"✅ {preset_name.upper()} dataset generated!")
    print(f"📁 Files: {students_file}, {internships_file}")
    print(f"📊 Stats: {len(students_df)} students, {len(internships_df)} internships")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick dataset generation')
    parser.add_argument('preset', choices=['small', 'medium', 'large', 'xlarge', 'test'], 
                       help='Dataset size preset')
    
    args = parser.parse_args()
    generate_preset(args.preset)