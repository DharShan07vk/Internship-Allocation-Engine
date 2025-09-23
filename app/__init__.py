"""Package initialization and setup"""
from app.core.config import settings
from app.core.database import create_tables

__version__ = "1.0.0"
__title__ = "Internship Allocation Engine"
__description__ = "AI-powered internship allocation system using ML and optimization"

# Initialize database tables
create_tables()
