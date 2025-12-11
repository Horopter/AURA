"""
DuckDB integration for fast analytics queries on pipeline data.

This module provides DuckDB-based analytics for querying metadata, features,
and training results across pipeline stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import polars as pl

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available. Install with: pip install duckdb")


class DuckDBAnalytics:
    """
    DuckDB-based analytics for pipeline data.
    
    Provides fast SQL queries on Polars DataFrames and Arrow/Parquet files.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB analytics.
        
        Args:
            db_path: Optional path to persistent DuckDB database
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not available. Install with: pip install duckdb")
        
        if db_path:
            self.conn = duckdb.connect(db_path)
        else:
            self.conn = duckdb.connect()
        
        logger.info("DuckDB analytics initialized")
    
    def register_dataframe(self, name: str, df: pl.DataFrame) -> None:
        """
        Register a Polars DataFrame as a DuckDB table.
        
        Args:
            name: Table name
            df: Polars DataFrame
        """
        self.conn.register(name, df)
        logger.debug(f"Registered DataFrame as table: {name} ({df.height} rows)")
    
    def register_arrow(self, name: str, arrow_path: str) -> None:
        """
        Register an Arrow file as a DuckDB table.
        
        Args:
            name: Table name
            arrow_path: Path to Arrow file
        """
        # DuckDB can read Arrow files directly via read_arrow or convert to Parquet
        # For Arrow IPC format, we need to read it with Polars first, then register
        try:
            df = pl.read_ipc(arrow_path)
            self.conn.register(name, df)
            logger.debug(f"Registered Arrow file as table: {name}")
        except Exception as e:
            logger.error(f"Failed to register Arrow file {arrow_path}: {e}")
            raise
    
    def register_parquet(self, name: str, parquet_path: str) -> None:
        """
        Register a Parquet file as a DuckDB table.
        
        Args:
            name: Table name
            parquet_path: Path to Parquet file
        """
        # Use parameterized query to prevent SQL injection
        # DuckDB's read_parquet requires the path to be a string literal in SQL
        # So we validate the path exists and is safe
        path_obj = Path(parquet_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        # Escape single quotes in path for SQL safety
        safe_path = str(parquet_path).replace("'", "''")
        self.conn.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{safe_path}')"
        )
        logger.debug(f"Registered Parquet file as table: {name}")
    
    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute a SQL query and return results as Polars DataFrame.
        
        Args:
            sql: SQL query string
        
        Returns:
            Polars DataFrame with query results
        """
        result = self.conn.execute(sql).pl()
        return result
    
    def get_video_statistics(self, metadata_table: str = "metadata") -> pl.DataFrame:
        """
        Get statistics about videos in the dataset.
        
        Args:
            metadata_table: Name of metadata table (must be a valid identifier)
        
        Returns:
            DataFrame with video statistics
        """
        # Validate table name to prevent SQL injection
        if not metadata_table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {metadata_table}")
        
        sql = f"""
        SELECT 
            label,
            COUNT(*) as video_count,
            COUNT(DISTINCT original_video) as unique_originals,
            AVG(CASE WHEN is_original THEN 1 ELSE 0 END) as original_ratio
        FROM {metadata_table}
        GROUP BY label
        """
        return self.query(sql)
    
    def get_augmentation_distribution(
        self, metadata_table: str = "metadata"
    ) -> pl.DataFrame:
        """
        Get distribution of augmentation types.
        
        Args:
            metadata_table: Name of metadata table (must be a valid identifier)
        
        Returns:
            DataFrame with augmentation distribution
        """
        # Validate table name to prevent SQL injection
        if not metadata_table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {metadata_table}")
        
        sql = f"""
        SELECT 
            augmentation_type,
            COUNT(*) as count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {metadata_table}) as percentage
        FROM {metadata_table}
        WHERE augmentation_type IS NOT NULL
        GROUP BY augmentation_type
        ORDER BY count DESC
        """
        return self.query(sql)
    
    def get_scaling_statistics(
        self, scaled_table: str = "scaled_metadata"
    ) -> pl.DataFrame:
        """
        Get statistics about video scaling.
        
        Args:
            scaled_table: Name of scaled metadata table (must be a valid identifier)
        
        Returns:
            DataFrame with scaling statistics
        """
        # Validate table name to prevent SQL injection
        if not scaled_table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {scaled_table}")
        
        sql = f"""
        SELECT 
            COUNT(*) as total_videos,
            SUM(CASE WHEN is_upscaled THEN 1 ELSE 0 END) as upscaled_count,
            SUM(CASE WHEN is_downscaled THEN 1 ELSE 0 END) as downscaled_count,
            AVG(scaled_width) as avg_width,
            AVG(scaled_height) as avg_height,
            AVG(original_width) as avg_original_width,
            AVG(original_height) as avg_original_height
        FROM {scaled_table}
        """
        return self.query(sql)
    
    def get_feature_statistics(
        self, features_table: str = "features"
    ) -> pl.DataFrame:
        """
        Get statistics about extracted features.
        
        Args:
            features_table: Name of features table (must be a valid identifier)
        
        Returns:
            DataFrame with feature statistics
        """
        # Validate table name to prevent SQL injection
        if not features_table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {features_table}")
        
        sql = f"""
        SELECT 
            COUNT(*) as total_samples,
            COUNT(DISTINCT video_path) as unique_videos,
            COUNT(DISTINCT label) as unique_labels
        FROM {features_table}
        """
        return self.query(sql)
    
    def get_training_results_summary(
        self, results_table: str = "training_results"
    ) -> pl.DataFrame:
        """
        Get summary of training results.
        
        Args:
            results_table: Name of training results table (must be a valid identifier)
        
        Returns:
            DataFrame with training results summary
        """
        # Validate table name to prevent SQL injection
        if not results_table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {results_table}")
        
        sql = f"""
        SELECT 
            model_type,
            AVG(val_acc) as avg_val_acc,
            STDDEV(val_acc) as std_val_acc,
            AVG(val_loss) as avg_val_loss,
            STDDEV(val_loss) as std_val_loss,
            COUNT(*) as fold_count
        FROM {results_table}
        GROUP BY model_type
        ORDER BY avg_val_acc DESC
        """
        return self.query(sql)
    
    def close(self) -> None:
        """Close DuckDB connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


__all__ = ["DuckDBAnalytics", "DUCKDB_AVAILABLE"]

