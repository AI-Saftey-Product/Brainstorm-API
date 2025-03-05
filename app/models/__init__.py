"""Database models for the application."""

from app.models.model import Model
from app.models.test_run import TestRun
from app.models.test_result import TestResult

# For Alembic migrations
from app.db.base import Base 