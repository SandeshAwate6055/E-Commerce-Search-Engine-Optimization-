"""Database writer supporting PostgreSQL with automatic SQLite fallback."""
import os
import sqlite3
from typing import List, Tuple, Optional
from backend.indexing.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseWriter:
    def __init__(self, config: dict, prefer_db: str = "postgres"):
        self.config = config
        self.prefer_db = prefer_db.lower()
        self.backend = None  # "postgres" or "sqlite"
        self.conn = None
        self.cursor = None
        self.table_name = config.get('table_name', 'products')

    def connect(self, db_type: str = None, sqlite_path: str = None):
        target = db_type or self.prefer_db

        if target == "postgres":
            try:
                import psycopg2
                self.conn = psycopg2.connect(
                    host=self.config.get('host', 'localhost'),
                    port=self.config.get('port', 5432),
                    dbname=self.config.get('dbname', 'ecommerce_search'),
                    user=self.config.get('user', 'postgres'),
                    password=self.config.get('password', 'postgres'),
                    connect_timeout=3
                )
                self.cursor = self.conn.cursor()
                self.backend = "postgres"
                logger.info("[OK] Connected to PostgreSQL database")
                return
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed ({e}). Falling back to SQLite.")

        # SQLite fallback / explicit selection
        sqlite_file = sqlite_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "processed", "products.db"
        )
        os.makedirs(os.path.dirname(os.path.abspath(sqlite_file)), exist_ok=True)
        self.conn = sqlite3.connect(sqlite_file)
        self.cursor = self.conn.cursor()
        self.backend = "sqlite"
        logger.info(f"[OK] Connected to SQLite database: {sqlite_file}")

    def create_table(self, schema_file: str = None):
        if self.backend == "postgres":
            with open(schema_file, 'r') as f:
                self.cursor.execute(f.read())
            self.conn.commit()
            logger.info("[OK] PostgreSQL table verified/created")
        else:
            query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pid              TEXT NOT NULL UNIQUE,
                image_path       TEXT NOT NULL,
                product_name     TEXT,
                main_category    TEXT,
                brand            TEXT,
                retail_price     REAL,
                discounted_price REAL,
                raw_caption      TEXT,
                norm_text        TEXT NOT NULL,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.cursor.execute(query)
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pid ON {self.table_name}(pid);")
            self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_category ON {self.table_name}(main_category);")
            self.conn.commit()
            logger.info("[OK] SQLite table verified/created")

    def insert_product(self, pid: str, image_path: str, product_name: str,
                       main_category: str, brand: str, retail_price: float,
                       discounted_price: float, raw_caption: str,
                       norm_text: str) -> Optional[int]:
        try:
            if self.backend == "postgres":
                query = f"""
                    INSERT INTO {self.table_name}
                        (pid, image_path, product_name, main_category, brand,
                         retail_price, discounted_price, raw_caption, norm_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pid) DO UPDATE
                    SET raw_caption = EXCLUDED.raw_caption,
                        norm_text = EXCLUDED.norm_text
                    RETURNING id
                """
                self.cursor.execute(query, (
                    pid, image_path, product_name, main_category, brand,
                    retail_price, discounted_price, raw_caption, norm_text
                ))
                row_id = self.cursor.fetchone()[0]
            else:
                query = f"""
                    INSERT INTO {self.table_name}
                        (pid, image_path, product_name, main_category, brand,
                         retail_price, discounted_price, raw_caption, norm_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (pid) DO UPDATE
                    SET raw_caption = excluded.raw_caption,
                        norm_text = excluded.norm_text
                """
                self.cursor.execute(query, (
                    pid, image_path, product_name, main_category, brand,
                    retail_price, discounted_price, raw_caption, norm_text
                ))
                row_id = self.cursor.lastrowid
            self.conn.commit()
            return row_id
        except Exception as e:
            logger.error(f"Insert error for PID {pid}: {e}")
            self.conn.rollback()
            return None

    def get_all_pids(self) -> set:
        self.cursor.execute(f"SELECT pid FROM {self.table_name}")
        return {row[0] for row in self.cursor.fetchall()}

    def count(self) -> int:
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return self.cursor.fetchone()[0]

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info(f"Database ({self.backend}) connection closed")
