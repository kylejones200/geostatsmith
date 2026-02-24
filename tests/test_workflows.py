"""
Tests for workflow and pipeline functionality
"""

import tempfile
from pathlib import Path

import pytest

from geostats.config import AnalysisConfig, ConfigError, load_config
from geostats.workflows.pipeline import AnalysisPipeline, PipelineError


class TestConfigLoading:
    """Test configuration loading"""

    def test_load_valid_config(self):
        """Test loading a valid config"""
        config_dict = {
            "project": {
                "name": "Test Analysis",
                "output_dir": "./test_output",
            },
            "data": {
                "input_file": "test_data.csv",
                "x_column": "x",
                "y_column": "y",
                "z_column": "z",
            },
        }
        config = AnalysisConfig(**config_dict)
        assert config.project.name == "Test Analysis"
        assert config.data.input_file == "test_data.csv"

    def test_load_config_from_yaml(self):
        """Test loading config from YAML file"""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "project": {
                        "name": "YAML Test",
                        "output_dir": "./yaml_output",
                    },
                    "data": {
                        "input_file": "data.csv",
                        "x_column": "x",
                        "y_column": "y",
                        "z_column": "z",
                    },
                },
                f,
            )
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config.project.name == "YAML Test"
        finally:
            Path(config_path).unlink()

    def test_invalid_config(self):
        """Test handling of invalid config"""
        with pytest.raises(ConfigError):
            load_config("nonexistent.yaml")

    def test_config_validation(self):
        """Test config validation"""
        # Missing required fields
        with pytest.raises(Exception):  # Pydantic validation error
            AnalysisConfig(project={"name": "Test"})  # Missing data


class TestPipeline:
    """Test analysis pipeline"""

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        config_dict = {
            "project": {
                "name": "Test Pipeline",
                "output_dir": "./test_output",
            },
            "data": {
                "input_file": "test_data.csv",
                "x_column": "x",
                "y_column": "y",
                "z_column": "z",
            },
        }
        config = AnalysisConfig(**config_dict)
        pipeline = AnalysisPipeline(config)
        assert pipeline.config == config
        assert pipeline.output_dir.exists()

    def test_pipeline_with_missing_data(self):
        """Test pipeline with missing data file"""
        config_dict = {
            "project": {
                "name": "Test Pipeline",
                "output_dir": "./test_output",
            },
            "data": {
                "input_file": "nonexistent.csv",
                "x_column": "x",
                "y_column": "y",
                "z_column": "z",
            },
        }
        config = AnalysisConfig(**config_dict)
        pipeline = AnalysisPipeline(config)

        # Should raise error when trying to load data
        with pytest.raises(PipelineError):
            pipeline.load_data()
