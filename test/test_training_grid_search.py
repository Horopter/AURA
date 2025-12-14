"""
Comprehensive unit tests for training/grid_search module.
Tests hyperparameter grid generation and combination.
"""
import pytest
from lib.training.grid_search import (
    get_hyperparameter_grid,
    generate_parameter_combinations,
    select_best_hyperparameters,
)


class TestGetHyperparameterGrid:
    """Tests for get_hyperparameter_grid function."""
    
    def test_get_hyperparameter_grid_logistic_regression(self):
        """Test get_hyperparameter_grid for logistic_regression."""
        grid = get_hyperparameter_grid("logistic_regression")
        assert isinstance(grid, dict)
        assert "C" in grid
        assert "max_iter" in grid
        assert isinstance(grid["C"], list)
        assert len(grid["C"]) > 0
    
    def test_get_hyperparameter_grid_svm(self):
        """Test get_hyperparameter_grid for svm."""
        grid = get_hyperparameter_grid("svm")
        assert isinstance(grid, dict)
        assert "C" in grid
        assert "kernel" in grid
    
    def test_get_hyperparameter_grid_video_model(self):
        """Test get_hyperparameter_grid for video model."""
        grid = get_hyperparameter_grid("slowfast")
        assert isinstance(grid, dict)
        assert "learning_rate" in grid
        assert "weight_decay" in grid
    
    def test_get_hyperparameter_grid_xgboost(self):
        """Test get_hyperparameter_grid for xgboost model."""
        grid = get_hyperparameter_grid("xgboost_pretrained_inception")
        assert isinstance(grid, dict)
        assert len(grid) > 0
    
    def test_get_hyperparameter_grid_invalid(self):
        """Test get_hyperparameter_grid with invalid model type."""
        grid = get_hyperparameter_grid("invalid_model")
        # Should return empty dict or default
        assert isinstance(grid, dict)


class TestGenerateParameterCombinations:
    """Tests for generate_parameter_combinations function."""
    
    def test_generate_parameter_combinations_basic(self):
        """Test generate_parameter_combinations with basic grid."""
        grid = {
            "param1": [1, 2],
            "param2": ["a", "b"]
        }
        combinations = generate_parameter_combinations(grid)
        
        assert isinstance(combinations, list)
        assert len(combinations) == 4  # 2 * 2 = 4 combinations
        assert all(isinstance(c, dict) for c in combinations)
        assert all("param1" in c and "param2" in c for c in combinations)
    
    def test_generate_parameter_combinations_single_param(self):
        """Test generate_parameter_combinations with single parameter."""
        grid = {"param1": [1, 2, 3]}
        combinations = generate_parameter_combinations(grid)
        
        assert len(combinations) == 3
        assert all(c["param1"] in [1, 2, 3] for c in combinations)
    
    def test_generate_parameter_combinations_empty(self):
        """Test generate_parameter_combinations with empty grid."""
        grid = {}
        combinations = generate_parameter_combinations(grid)
        
        assert isinstance(combinations, list)
        assert len(combinations) == 1  # One empty dict
        assert combinations[0] == {}


class TestSelectBestHyperparameters:
    """Tests for select_best_hyperparameters function."""
    
    def test_select_best_hyperparameters_basic(self):
        """Test select_best_hyperparameters with basic results."""
        results = [
            {"params": {"C": 0.1}, "score": 0.8},
            {"params": {"C": 1.0}, "score": 0.9},
            {"params": {"C": 10.0}, "score": 0.85},
        ]
        
        best = select_best_hyperparameters(results)
        
        assert best == {"C": 1.0}  # Highest score
    
    def test_select_best_hyperparameters_empty(self):
        """Test select_best_hyperparameters with empty results."""
        results = []
        best = select_best_hyperparameters(results)
        
        assert best is None or best == {}
    
    def test_select_best_hyperparameters_tie(self):
        """Test select_best_hyperparameters with tied scores."""
        results = [
            {"params": {"C": 0.1}, "score": 0.9},
            {"params": {"C": 1.0}, "score": 0.9},
        ]
        
        best = select_best_hyperparameters(results)
        # Should return one of them
        assert best in [{"C": 0.1}, {"C": 1.0}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

