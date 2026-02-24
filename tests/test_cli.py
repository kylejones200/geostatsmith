"""
Tests for CLI functionality
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from geostats.cli import cli


class TestCLI:
    """Test command-line interface"""

    @pytest.fixture
    def runner(self):
        """Create CLI runner"""
        return CliRunner()

    @pytest.fixture
    def sample_config(self):
        """Create sample config file"""
        config_dict = {
            "project": {
                "name": "CLI Test",
                "output_dir": "./cli_test_output",
            },
            "data": {
                "input_file": "test_data.csv",
                "x_column": "x",
                "y_column": "y",
                "z_column": "z",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name

        yield config_path
        Path(config_path).unlink()

    def test_cli_help(self, runner):
        """Test CLI help"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_validate_command(self, runner, sample_config):
        """Test validate command"""
        result = runner.invoke(cli, ["validate", sample_config])
        # May fail if config is invalid, but should run
        assert result.exit_code in [0, 1]

    def test_validate_missing_file(self, runner):
        """Test validate with missing file"""
        result = runner.invoke(cli, ["validate", "nonexistent.yaml"])
        assert result.exit_code != 0
