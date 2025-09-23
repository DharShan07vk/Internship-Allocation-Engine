import sys
from loguru import logger
from app.core.config import settings

def setup_logging():
    """Configure logging settings"""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stdout,
        format=settings.get("LOG_FORMAT", "{time} | {level} | {message}"),
        level=settings.get("LOG_LEVEL", "INFO"),
        colorize=True
    )
    
    # File logging
    logger.add(
        "logs/app.log",
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days"
    )
    
    return logger

# Initialize logging
setup_logging()
