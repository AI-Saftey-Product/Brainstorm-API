from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# Setup database connection only if not disabled
if not settings.DISABLE_DATABASE:
    try:
        SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Dependency to get DB session
        def get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()
                
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Using mock database.")
        # Fall back to mock implementation if connection fails
        settings.DISABLE_DATABASE = True

# Mock database implementation when database is disabled
if settings.DISABLE_DATABASE:
    logger.info("Using mock database implementation")
    
    class MockSession:
        """Mock session that doesn't perform any actual database operations."""
        def __init__(self):
            self.items = {}
            
        def add(self, obj):
            # Just pretend to add items
            pass
            
        def commit(self):
            # Do nothing
            pass
            
        def refresh(self, obj):
            # Do nothing
            pass
            
        def rollback(self):
            # Do nothing
            pass
            
        def close(self):
            # Do nothing
            pass
            
        def query(self, *args, **kwargs):
            # Return empty query
            return MockQuery()
    
    class MockQuery:
        """Mock query that returns empty results."""
        def filter(self, *args, **kwargs):
            return self
            
        def all(self):
            return []
            
        def first(self):
            return None
    
    # Define get_db to return mock session
    @contextmanager
    def get_db_context():
        yield MockSession()
    
    def get_db():
        """Dependency to get mock DB session"""
        yield MockSession() 