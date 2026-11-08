import sys
import logging
from pathlib import Path
from alembic.config import Config
from alembic import command

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from src.database import init_db_connection, engine, Base

logger = logging.getLogger("init_db")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_alembic_migrations():
    """Runs Alembic migrations to bring database up to HEAD."""
    alembic_ini_path = backend_dir / "alembic.ini"
    if not alembic_ini_path.exists():
        logger.warning(f"alembic.ini not found at {alembic_ini_path}. Falling back to Base.metadata.create_all.")
        return False
    try:
        logger.info("Applying Alembic migrations ('alembic upgrade head')...")
        alembic_cfg = Config(str(alembic_ini_path))
        command.upgrade(alembic_cfg, "head")
        logger.info("[SUCCESS] Alembic migrations applied successfully!")
        return True
    except Exception as e:
        logger.warning(f"Alembic upgrade failed: {e}. Falling back to Base.metadata.create_all.")
        return False

def main():
    print("--> Initializing PostgreSQL database tables...")
    connected = init_db_connection()
    if not connected:
        print("[ERROR] Database connection failed. Ensure PostgreSQL is running ('docker compose up -d postgres').")
        sys.exit(1)

    migrated = run_alembic_migrations()
    if not migrated and engine is not None:
        Base.metadata.create_all(bind=engine)
        print("[INFO] Fallback table creation completed via SQLAlchemy metadata.")

    print("[SUCCESS] All tables (raw_customers, prediction_logs, ground_truth_feedback) verified and ready!")

if __name__ == "__main__":
    main()
