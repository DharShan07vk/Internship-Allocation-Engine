"""
Quick Setup and Installation Script for Internship Allocation Engine
=================================================================

This script automates the complete setup process including:
1. Environment setup and dependency installation
2. Data generation
3. Initial model training
4. System validation

Usage:
    python setup.py --quick-setup
    python setup.py --full-setup --large-dataset
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a command with proper error handling"""
    print(f"\n🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ❌ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def setup_virtual_environment():
    """Create and setup virtual environment"""
    print("\n🌍 Setting up virtual environment...")
    
    venv_path = "venv"
    if not os.path.exists(venv_path):
        success = run_command("python -m venv venv", "Creating virtual environment")
        if not success:
            return False
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    success = run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    if not success:
        print("⚠️  Pip upgrade failed, continuing...")
    
    # Install dependencies
    success = run_command(f"{pip_cmd} install -r requirements.txt", 
                         "Installing dependencies")
    return success

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data", "models", "logs", "analytics", 
        "tests", "docs", "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   📁 {directory}/")
    
    print("✅ Directory structure created")
    return True

def generate_sample_data(scale="medium"):
    """Generate sample data for testing"""
    print(f"\n📊 Generating {scale} sample dataset...")
    
    success = run_command(f"python data/generate.py --scale {scale} --validate", 
                         f"Generating {scale} dataset")
    return success

def run_initial_training():
    """Run initial model training"""
    print("\n🤖 Running initial model training...")
    
    success = run_command("python runPython.py --train-model --force-embeddings --top-k-sim 20 --top-k-opt 15", 
                         "Training initial models")
    return success

def validate_installation():
    """Validate the installation"""
    print("\n✅ Validating installation...")
    
    # Check if key files exist
    key_files = [
        "app/main.py",
        "app/services/similarity.py", 
        "app/services/prediction.py",
        "app/services/optimizer.py",
        "data/students.csv",
        "data/internships.csv",
        "requirements.txt"
    ]
    
    all_good = True
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            all_good = False
    
    return all_good

def quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test...")
    
    test_script = """
import sys
sys.path.append('.')

try:
    # Test imports
    from app.services.similarity import load_data
    from app.services.prediction import SuccessPredictionModel
    from app.services.optimizer import AllocationOptimizer
    
    # Test data loading
    students_df, internships_df = load_data()
    print(f"✅ Data loading: {len(students_df)} students, {len(internships_df)} internships")
    
    # Test similarity engine
    from app.services.similarity import similarity_engine
    print("✅ Similarity engine initialized")
    
    # Test prediction model
    model = SuccessPredictionModel()
    print("✅ Prediction model initialized")
    
    # Test optimizer
    optimizer = AllocationOptimizer()
    print("✅ Optimizer initialized")
    
    print("🎉 All core components working!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
    """
    
    with open("temp/quick_test.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python temp/quick_test.py", "Running component tests")
    
    # Cleanup
    if os.path.exists("temp/quick_test.py"):
        os.remove("temp/quick_test.py")
    
    return success

def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(
        description="Setup script for Internship Allocation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--quick-setup", action="store_true",
                       help="Quick setup with small dataset")
    parser.add_argument("--full-setup", action="store_true", 
                       help="Full setup with model training")
    parser.add_argument("--large-dataset", action="store_true",
                       help="Generate large dataset (use with --full-setup)")
    parser.add_argument("--skip-venv", action="store_true",
                       help="Skip virtual environment setup")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip initial model training")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run validation tests")
    
    args = parser.parse_args()
    
    if not any([args.quick_setup, args.full_setup, args.test_only]):
        parser.print_help()
        return 1
    
    print("🚀 Internship Allocation Engine Setup")
    print("=" * 50)
    
    start_time = time.time()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    if args.test_only:
        success = validate_installation() and quick_test()
        print(f"\n{'✅ Validation passed!' if success else '❌ Validation failed!'}")
        return 0 if success else 1
    
    # Setup virtual environment
    if not args.skip_venv:
        if not setup_virtual_environment():
            print("❌ Virtual environment setup failed")
            return 1
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return 1
    
    # Generate data
    dataset_scale = "large" if args.large_dataset else "medium" if args.full_setup else "small"
    if not generate_sample_data(dataset_scale):
        print("❌ Data generation failed")
        return 1
    
    # Initial training (for full setup)
    if args.full_setup and not args.skip_training:
        if not run_initial_training():
            print("❌ Initial training failed")
            return 1
    
    # Validation
    if not validate_installation():
        print("❌ Installation validation failed")
        return 1
    
    # Quick test
    if not quick_test():
        print("❌ Functionality test failed")
        return 1
    
    # Success summary
    setup_time = time.time() - start_time
    print(f"\n🎉 Setup completed successfully in {setup_time:.1f} seconds!")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Activate virtual environment:")
    if os.name == 'nt':
        print(f"      venv\\Scripts\\activate")
    else:
        print(f"      source venv/bin/activate")
    
    print(f"   2. Run the allocation pipeline:")
    print(f"      python runPython.py --train-model")
    
    print(f"   3. Start the API server:")
    print(f"      python -m uvicorn app.main:app --reload")
    
    print(f"   4. Access the web interface:")
    print(f"      http://localhost:8000")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
