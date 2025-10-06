"""Database connection and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from src.database.models import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, db_url: str = None):
        """
        Initialize database manager.

        Args:
            db_url: Database URL (default: SQLite)
        """
        if db_url is None:
            # Default to SQLite for development
            db_path = os.path.join("data", "dementia_chatbot.db")
            os.makedirs("data", exist_ok=True)
            db_url = f"sqlite:///{db_path}"

        # Create engine
        if db_url.startswith("sqlite"):
            # SQLite-specific configuration
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
        else:
            # PostgreSQL or other databases
            self.engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )

        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )

    def create_all_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_all_tables(self):
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator:
        """
        Get database session with automatic cleanup.

        Yields:
            Database session

        Example:
            with db_manager.get_session() as session:
                user = session.query(UserProfile).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def close(self):
        """Close all database connections."""
        self.SessionLocal.remove()
        self.engine.dispose()


# Global database manager instance
_db_manager = None


def get_db_manager(db_url: str = None) -> DatabaseManager:
    """
    Get or create global database manager instance.

    Args:
        db_url: Database URL (optional)

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_url)
    return _db_manager


def init_database(db_url: str = None, reset: bool = False):
    """
    Initialize database with tables.

    Args:
        db_url: Database URL (optional)
        reset: If True, drop existing tables first
    """
    db_manager = get_db_manager(db_url)

    if reset:
        print("Dropping existing tables...")
        db_manager.drop_all_tables()

    print("Creating database tables...")
    db_manager.create_all_tables()
    print("Database initialized successfully!")


if __name__ == "__main__":
    # Initialize database when run directly
    init_database(reset=False)
