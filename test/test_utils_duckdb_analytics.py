"""
Comprehensive unit tests for utils/duckdb_analytics module.
Tests DuckDB analytics with dummy data.
"""
import pytest
import polars as pl
from pathlib import Path
from lib.utils.duckdb_analytics import DuckDBAnalytics


class TestDuckDBAnalytics:
    """Tests for DuckDBAnalytics class."""
    
    def test_initialization(self):
        """Test DuckDBAnalytics initialization."""
        try:
            analytics = DuckDBAnalytics()
            assert analytics.conn is not None
        except ImportError:
            pytest.skip("DuckDB not available")
    
    def test_register_dataframe(self):
        """Test register_dataframe method."""
        try:
            analytics = DuckDBAnalytics()
            df = pl.DataFrame({
                "video_path": ["video1.mp4", "video2.mp4"],
                "label": [0, 1]
            })
            
            analytics.register_dataframe("test_table", df)
            # Should not raise
        except ImportError:
            pytest.skip("DuckDB not available")
    
    def test_query(self):
        """Test query method."""
        try:
            analytics = DuckDBAnalytics()
            df = pl.DataFrame({
                "video_path": ["video1.mp4", "video2.mp4"],
                "label": [0, 1]
            })
            analytics.register_dataframe("test_table", df)
            
            result = analytics.query("SELECT * FROM test_table WHERE label = 0")
            assert isinstance(result, pl.DataFrame)
            assert result.height == 1
        except ImportError:
            pytest.skip("DuckDB not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
