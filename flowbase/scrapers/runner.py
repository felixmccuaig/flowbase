"""Scraper runner for executing scheduled data collection."""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
import yaml

from flowbase.tables.manager import TableManager


class ScraperRunner:
    """Runs scrapers and ingests results into tables."""

    def __init__(self, metadata_db: Optional[str] = None):
        """Initialize scraper runner.

        Args:
            metadata_db: Path to metadata database (defaults to data/tables/.metadata.db)
        """
        if metadata_db is None:
            metadata_db = "data/tables/.metadata.db"
        self.table_manager = TableManager(metadata_db=metadata_db)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load scraper configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_scraper_function(self, function_path: str) -> Callable:
        """Load a scraper function from a Python file.

        Args:
            function_path: Path in format "path/to/file.py:function_name"

        Returns:
            The scraper function
        """
        # Parse the path
        if ':' not in function_path:
            raise ValueError(f"Function path must be in format 'file.py:function_name', got: {function_path}")

        file_path, function_name = function_path.rsplit(':', 1)
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Scraper file not found: {file_path}")

        # Load the module
        spec = importlib.util.spec_from_file_location("scraper_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["scraper_module"] = module
        spec.loader.exec_module(module)

        # Get the function
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in {file_path}")

        return getattr(module, function_name)

    def run(self, config_path: str, date: Optional[str] = None, **kwargs) -> None:
        """Run a scraper and ingest results.

        Args:
            config_path: Path to scraper YAML config
            date: Date to scrape (YYYY-MM-DD), defaults to today
            **kwargs: Additional parameters to pass to scraper function
        """
        config = self.load_config(config_path)

        # Parse date
        if date:
            scrape_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            scrape_date = datetime.now()

        # Load scraper function
        scraper_fn = self.load_scraper_function(config["scraper"]["function"])

        # Merge config parameters with runtime kwargs
        params = config["scraper"].get("parameters", {}).copy()
        params.update(kwargs)

        print(f"Running scraper: {config['name']}")
        print(f"Date: {scrape_date.strftime('%Y-%m-%d')}")
        print(f"Function: {config['scraper']['function']}")

        # Run the scraper
        result_df = scraper_fn(date=scrape_date, **params)

        if not isinstance(result_df, pd.DataFrame):
            raise TypeError(f"Scraper must return a pandas DataFrame, got {type(result_df)}")

        print(f"Scraped {len(result_df)} rows")

        # Get output configuration
        output_config = config["output"]
        table_config_path = output_config["table_config"]

        # Save to temporary parquet file
        temp_dir = Path("data/scrapers/.temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"{config['name']}_{scrape_date.strftime('%Y%m%d')}.parquet"
        result_df.to_parquet(temp_file, index=False)

        # Ingest into table
        print(f"Ingesting into table: {output_config['table']}")
        self.table_manager.ingest(
            table_name=output_config["table"],
            config_path=table_config_path,
            source_file=str(temp_file),
            date=scrape_date.strftime("%Y-%m-%d"),
            dataset_config_path=output_config.get("dataset_config")
        )

        # Clean up temp file
        temp_file.unlink()

        print("✓ Scraper completed successfully")

    def list_scrapers(self, scrapers_dir: str = "scrapers") -> list:
        """List all available scraper configs.

        Args:
            scrapers_dir: Directory containing scraper configs

        Returns:
            List of scraper config paths
        """
        scrapers_path = Path(scrapers_dir)
        if not scrapers_path.exists():
            return []

        return [str(p) for p in scrapers_path.rglob("*_scraper.yaml")]
