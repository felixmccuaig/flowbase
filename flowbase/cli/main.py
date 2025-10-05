"""Main CLI entry point."""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from flowbase.core.config.loader import load_config, load_pipeline_config, load_experiment_config
from flowbase.pipelines.executor import PipelineExecutor
from flowbase.experiments.runner import ExperimentRunner
from flowbase.experiments.tracker import ExperimentTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Flowbase - Tabular ML platform without infrastructure complexity."""
    pass


@cli.command()
@click.argument("project_name")
@click.option("--path", default=".", help="Directory to create project in")
def init(project_name: str, path: str) -> None:
    """Initialize a new Flowbase project."""
    project_path = Path(path) / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (project_path / "data").mkdir(exist_ok=True)
    (project_path / "pipelines").mkdir(exist_ok=True)
    (project_path / "experiments").mkdir(exist_ok=True)
    (project_path / "models").mkdir(exist_ok=True)

    # Create example config
    config_content = f"""project_name: {project_name}
version: 1.0.0
storage: local
query_engine: duckdb

pipelines: []
experiments: []
"""

    config_file = project_path / "flowbase.yaml"
    config_file.write_text(config_content)

    # Create example pipeline
    pipeline_content = """name: example_pipeline
description: Example data pipeline

sources:
  - name: raw_data
    type: file
    path: data/input.csv
    format: csv

features:
  - name: basic_features
    sql: |
      SELECT
        *,
        column1 * 2 as feature1,
        column2 + 100 as feature2
      FROM raw_data
    sources:
      - raw_data
    materialize: true

output_path: data/features
"""

    pipeline_file = project_path / "pipelines" / "example.yaml"
    pipeline_file.write_text(pipeline_content)

    # Create example experiment
    experiment_content = """name: example_experiment
description: Example ML experiment

dataset: basic_features

models:
  - name: logistic_regression
    type: sklearn
    class_name: linear_model.LogisticRegression
    hyperparameters:
      max_iter: 1000
      random_state: 42
    features:
      - feature1
      - feature2
    target: target

  - name: random_forest
    type: sklearn
    class_name: ensemble.RandomForestClassifier
    hyperparameters:
      n_estimators: 100
      random_state: 42
    features:
      - feature1
      - feature2
    target: target

metrics:
  - accuracy
  - f1
  - roc_auc

test_size: 0.2
random_state: 42
"""

    experiment_file = project_path / "experiments" / "example.yaml"
    experiment_file.write_text(experiment_content)

    console.print(f"[green]✓[/green] Created Flowbase project: {project_name}")
    console.print(f"\nNext steps:")
    console.print(f"  cd {project_name}")
    console.print(f"  # Add your data to data/")
    console.print(f"  flowbase pipeline run pipelines/example.yaml")
    console.print(f"  flowbase experiment run experiments/example.yaml")


@cli.group()
def dataset() -> None:
    """Manage datasets (cleaned, typed data)."""
    pass


@dataset.command("compile")
@click.argument("config_file")
@click.argument("source_file", required=False)
@click.option("--output", "-o", help="Output parquet file")
@click.option("--preview", is_flag=True, help="Show preview of results")
def dataset_compile(config_file: str, source_file: str, output: str, preview: bool) -> None:
    """Compile a dataset from source data."""
    from flowbase.pipelines.dataset_compiler import DatasetCompiler, load_dataset_config
    from flowbase.query.engines.duckdb_engine import DuckDBEngine
    import yaml
    from pathlib import Path

    try:
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle wrapped or unwrapped config
        if 'dataset' in config:
            dataset_config = config['dataset']
        else:
            dataset_config = config

        console.print(f"[blue]Compiling dataset:[/blue] {dataset_config['name']}")

        # Compile to SQL
        compiler = DatasetCompiler(source_table="raw_data")
        sql = compiler.compile(dataset_config)

        console.print("[dim]Generated SQL:[/dim]")
        console.print(sql)
        console.print()

        # Execute
        with console.status("[bold blue]Executing..."):
            engine = DuckDBEngine()

            # Check if merged dataset (multiple sources)
            sources = dataset_config.get("sources")
            if sources:
                # Register each source dataset
                for source_cfg in sources:
                    source_name = source_cfg['name']
                    dataset_config_path = source_cfg.get('dataset_config')

                    if dataset_config_path:
                        # Load and compile the source dataset
                        source_config = load_dataset_config(dataset_config_path)
                        source_sql = compiler.compile(source_config['dataset'])

                        # Register source table from config
                        table_config_path = source_config['dataset']['source'].get('table_config')
                        if table_config_path:
                            with open(table_config_path, 'r') as f:
                                table_config = yaml.safe_load(f)
                            table_name = table_config['table']['name']
                            base_path = table_config['table']['destination']['base_path']
                            file_format = table_config['table']['destination']['file_format']
                            pattern = f"{base_path}/*.{file_format}"

                            console.print(f"[dim]Registering source '{table_name}' from {pattern}[/dim]")
                            engine.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{pattern}'")

                        # Create view for this source dataset
                        console.print(f"[dim]Creating source view '{source_name}'[/dim]")
                        engine.execute(f"CREATE OR REPLACE VIEW {source_name} AS {source_sql}")
            else:
                # Single source - register file
                if source_file:
                    path = Path(source_file)
                    file_format = "parquet" if path.suffix == ".parquet" else "csv"
                    engine.register_file("raw_data", source_file, file_format)
                else:
                    # Try to get from config
                    source = dataset_config.get('source', {})
                    if source.get('table_config'):
                        with open(source['table_config'], 'r') as f:
                            table_config = yaml.safe_load(f)
                        table_name = table_config['table']['name']
                        base_path = table_config['table']['destination']['base_path']
                        file_format = table_config['table']['destination']['file_format']
                        pattern = f"{base_path}/*.{file_format}"

                        console.print(f"[dim]Registering table '{table_name}' from {pattern}[/dim]")
                        engine.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{pattern}'")

            # Execute
            result_df = engine.execute(sql)

        console.print(f"[green]✓[/green] Generated {len(result_df):,} rows × {len(result_df.columns)} columns")

        # Preview
        if preview:
            console.print("\n[bold]Preview:[/bold]")
            console.print(result_df.head(10).to_string())

        # Save
        if output:
            result_df.to_parquet(output)
            console.print(f"[green]✓[/green] Saved to {output}")
        else:
            # Default output location
            output = f"data/datasets/{dataset_config['name']}.parquet"
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_parquet(output)
            console.print(f"[green]✓[/green] Saved to {output}")

        engine.close()

    except Exception as e:
        console.print(f"[red]✗[/red] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def features() -> None:
    """Manage feature sets."""
    pass


@features.command("compile")
@click.argument("config_file")
@click.option("--dataset", "-d", help="Dataset file (parquet) or use raw source")
@click.option("--output", "-o", help="Output parquet file")
@click.option("--preview", is_flag=True, help="Show preview of results")
def features_compile(config_file: str, dataset: str, output: str, preview: bool) -> None:
    """Compile a feature set from a dataset."""
    from flowbase.pipelines.feature_compiler import FeatureCompiler
    from flowbase.query.engines.duckdb_engine import DuckDBEngine
    import yaml
    from pathlib import Path

    try:
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle wrapped config
        if 'features' in config:
            config = config['features']

        console.print(f"[blue]Compiling features:[/blue] {config['name']}")

        # Determine source table
        dataset_ref = config.get("dataset")
        if dataset_ref:
            source_table = dataset_ref
        else:
            source_table = "raw_data"

        # Compile to SQL
        compiler = FeatureCompiler(source_table=source_table)
        sql = compiler.compile(config)

        console.print("[dim]Generated SQL:[/dim]")
        console.print(sql)
        console.print()

        # Execute
        with console.status("[bold blue]Executing..."):
            engine = DuckDBEngine()

            # Register source
            if dataset:
                path = Path(dataset)
                file_format = "parquet" if path.suffix == ".parquet" else "csv"
                table_name = dataset_ref if dataset_ref else "raw_data"
                engine.register_file(table_name, dataset, file_format)
            else:
                console.print("[yellow]Warning:[/yellow] No dataset specified, using raw_data")

            # Execute
            result_df = engine.execute(sql)

        console.print(f"[green]✓[/green] Generated {len(result_df):,} rows × {len(result_df.columns)} columns")

        # Preview
        if preview:
            console.print("\n[bold]Preview:[/bold]")
            console.print(result_df.head(10).to_string())

        # Save
        if output:
            result_df.to_parquet(output)
            console.print(f"[green]✓[/green] Saved to {output}")
        else:
            # Default output location
            output = f"data/features/{config['name']}.parquet"
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_parquet(output)
            console.print(f"[green]✓[/green] Saved to {output}")

        engine.close()

    except Exception as e:
        console.print(f"[red]✗[/red] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def model() -> None:
    """Manage ML models."""
    pass


@model.command("train")
@click.argument("config_file")
@click.option("--features", "-f", required=True, help="Feature parquet file")
@click.option("--output", "-o", help="Output directory for model")
def model_train(config_file: str, features: str, output: str) -> None:
    """Train a model from config."""
    from flowbase.models.trainer import ModelTrainer
    import yaml
    from pathlib import Path

    try:
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Handle wrapped config
        if 'model' in config:
            config = config['model']

        console.print(f"[blue]Training model:[/blue] {config['name']}")

        # Train
        models_dir = output or "data/models"
        trainer = ModelTrainer(models_dir=models_dir)

        with console.status("[bold blue]Training..."):
            result = trainer.train(config, features)

        console.print(f"[green]✓[/green] Model trained successfully")
        console.print(f"\n[bold]Training Results:[/bold]")
        console.print(f"  Train size: {result['train_size']:,} samples")
        console.print(f"  Test size: {result['test_size']:,} samples")
        console.print(f"\n[bold]Metrics:[/bold]")
        for metric, value in result['metrics'].items():
            console.print(f"  {metric}: {value:.4f}")

        console.print(f"\n[dim]Model saved to:[/dim] {result['model_path']}")
        console.print(f"[dim]Metadata saved to:[/dim] {result['metadata_path']}")

    except Exception as e:
        console.print(f"[red]✗[/red] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@model.command("predict")
@click.argument("model_name")
@click.option("--input", "-i", "input_json", required=True, help="JSON string or file path with input features")
@click.option("--models-dir", "-d", default="data/models", help="Directory containing models")
def model_predict(model_name: str, input_json: str, models_dir: str) -> None:
    """Make a prediction using a trained model."""
    from flowbase.models.trainer import ModelTrainer
    import json
    from pathlib import Path

    try:
        # Parse input
        if Path(input_json).exists():
            with open(input_json, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(input_json)

        console.print(f"[blue]Making prediction with:[/blue] {model_name}\n")
        console.print("[dim]Input features:[/dim]")
        for key, value in input_data.items():
            console.print(f"  {key}: {value}")
        console.print()

        # Make prediction
        trainer = ModelTrainer(models_dir=models_dir)
        result = trainer.predict(model_name, input_data)

        console.print("[green]✓[/green] Prediction complete\n")
        console.print(f"[bold]Prediction:[/bold] {result['prediction']}")

        # Show probabilities if available
        if "probabilities" in result:
            console.print("\n[bold]Class Probabilities:[/bold]")
            for cls, prob in result["probabilities"].items():
                console.print(f"  Class {cls}: {prob:.4f} ({prob*100:.2f}%)")

    except Exception as e:
        console.print(f"[red]✗[/red] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def eval() -> None:
    """Manage model evaluations."""
    pass


@eval.command("compare")
@click.argument("model_paths", nargs=-1, required=True)
@click.option("--name", "-n", help="Evaluation name")
def eval_compare(model_paths: tuple, name: str) -> None:
    """Compare multiple trained models."""
    import json
    from pathlib import Path

    try:
        eval_name = name or "model_comparison"
        console.print(f"[blue]Comparing models:[/blue] {eval_name}\n")

        # Load all model metadata
        models_data = []
        for model_path in model_paths:
            metadata_path = Path(model_path).with_name(
                Path(model_path).stem + "_metadata.json"
            )
            if not metadata_path.exists():
                console.print(f"[yellow]Warning:[/yellow] Metadata not found for {model_path}")
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                models_data.append(metadata)

        if not models_data:
            console.print("[red]✗[/red] No valid models found")
            sys.exit(1)

        # Create comparison table
        # Detect if regression or classification based on first model
        is_regression = "mse" in models_data[0]["metrics"]

        table = Table(title=f"Model Comparison: {eval_name}")
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Test Score", style="green")

        if is_regression:
            table.add_column("RMSE", style="green")
            table.add_column("MAE", style="green")
            table.add_column("R²", style="green")
        else:
            table.add_column("Accuracy", style="green")
            table.add_column("F1", style="green")

        table.add_column("Train Size", style="blue")
        table.add_column("Test Size", style="blue")

        best_model = None
        best_score = -1

        for model in models_data:
            metrics = model["metrics"]
            test_score = metrics.get("test_score", 0)

            if test_score > best_score:
                best_score = test_score
                best_model = model["name"]

            if is_regression:
                table.add_row(
                    model["name"],
                    model["model_type"],
                    f"{test_score:.4f}",
                    f"{metrics.get('rmse', 0):,.0f}",
                    f"{metrics.get('mae', 0):,.0f}",
                    f"{metrics.get('r2', 0):.4f}",
                    f"{model['train_size']:,}",
                    f"{model['test_size']:,}"
                )
            else:
                table.add_row(
                    model["name"],
                    model["model_type"],
                    f"{test_score:.4f}",
                    f"{metrics.get('accuracy', 0):.4f}",
                    f"{metrics.get('f1', 0):.4f}",
                    f"{model['train_size']:,}",
                    f"{model['test_size']:,}"
                )

        console.print(table)
        console.print(f"\n[bold green]Best model:[/bold green] {best_model} (test_score: {best_score:.4f})")

        # Save comparison results
        comparison_dir = Path("data/evals")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = comparison_dir / f"{eval_name}.json"
        with open(comparison_path, 'w') as f:
            json.dump({
                "name": eval_name,
                "models": models_data,
                "best_model": best_model,
                "best_score": best_score
            }, f, indent=2)

        console.print(f"\n[dim]Comparison saved to:[/dim] {comparison_path}")

    except Exception as e:
        console.print(f"[red]✗[/red] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def experiment() -> None:
    """Manage ML experiments."""
    pass


@experiment.command("run")
@click.argument("config_file")
def experiment_run(config_file: str) -> None:
    """Run an ML experiment."""
    try:
        config = load_experiment_config(config_file)
        runner = ExperimentRunner()

        with console.status(f"[bold blue]Running experiment: {config.name}"):
            summary = runner.run(config)

        console.print(f"[green]✓[/green] Experiment completed: {config.name}")
        console.print(f"Trained {summary['total_runs']} model(s)")

        # Display results table
        table = Table(title="Model Results")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Metrics", style="green")

        for result in summary["results"]:
            status = "✓" if result["status"] == "completed" else "✗"
            metrics_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in result.get("metrics", {}).items()]
            )
            table.add_row(result["model_name"], status, metrics_str)

        console.print(table)

        if summary["best_run"]:
            console.print(f"\n[bold]Best model:[/bold] {summary['best_run']['run_name']}")

        runner.close()

    except Exception as e:
        console.print(f"[red]✗[/red] Experiment failed: {e}")
        sys.exit(1)


@experiment.command("list")
@click.option("--experiment", help="Filter by experiment name")
def experiment_list(experiment: str = None) -> None:
    """List experiments and runs."""
    tracker = ExperimentTracker()

    if experiment:
        # Show runs for specific experiment
        cursor = tracker.conn.execute(
            "SELECT id FROM experiments WHERE name = ?", (experiment,)
        )
        result = cursor.fetchone()

        if not result:
            console.print(f"[yellow]Experiment not found: {experiment}[/yellow]")
            return

        exp_id = result["id"]
        runs = tracker.get_experiment_runs(exp_id)

        table = Table(title=f"Runs for {experiment}")
        table.add_column("Run Name", style="cyan")
        table.add_column("Model Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Created", style="blue")

        for run in runs:
            table.add_row(
                run["run_name"],
                run["model_type"],
                run["status"],
                run["created_at"],
            )

        console.print(table)
    else:
        # List all experiments
        cursor = tracker.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        experiments = [dict(row) for row in cursor.fetchall()]

        table = Table(title="Experiments")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Created", style="blue")

        for exp in experiments:
            table.add_row(
                exp["name"],
                exp["description"] or "",
                exp["created_at"],
            )

        console.print(table)

    tracker.close()


@cli.group()
def table() -> None:
    """Manage tables (incremental loading and compaction)."""
    pass


@table.command("create")
@click.argument("config_file")
def table_create(config_file: str) -> None:
    """Create a new table from config."""
    from flowbase.tables.manager import TableManager

    try:
        manager = TableManager()
        result = manager.create_table(config_file)

        console.print(f"[green]✓[/green] Table created: {result['table_name']}")
        console.print(f"[dim]Base path:[/dim] {result['base_path']}")

        manager.close()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create table: {e}")
        sys.exit(1)


@table.command("ingest")
@click.argument("config_file")
@click.argument("source_file")
@click.option("--date", required=True, help="Date for this data (YYYY-MM-DD)")
@click.option("--dataset-config", help="Optional dataset cleaning config")
def table_ingest(config_file: str, source_file: str, date: str, dataset_config: str) -> None:
    """Ingest data for a specific date."""
    from flowbase.tables.manager import TableManager
    import yaml

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        table_name = config["name"]

        manager = TableManager()

        with console.status(f"[bold blue]Ingesting data for {date}..."):
            result = manager.ingest(
                table_name=table_name,
                config_path=config_file,
                source_file=source_file,
                date=date,
                dataset_config_path=dataset_config
            )

        console.print(f"[green]✓[/green] Ingested {result['rows']:,} rows")
        console.print(f"[dim]Destination:[/dim] {result['destination']}")

        manager.close()
    except Exception as e:
        console.print(f"[red]✗[/red] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@table.command("compact")
@click.argument("config_file")
@click.option("--period", required=True, help="Period to compact (YYYY-MM)")
@click.option("--delete-source", is_flag=True, help="Delete source files after compaction")
def table_compact(config_file: str, period: str, delete_source: bool) -> None:
    """Compact daily files into monthly files."""
    from flowbase.tables.manager import TableManager
    import yaml

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        table_name = config["name"]

        manager = TableManager()

        with console.status(f"[bold blue]Compacting {period}..."):
            result = manager.compact(
                table_name=table_name,
                config_path=config_file,
                period=period,
                delete_source=delete_source
            )

        if result["status"] == "skipped":
            console.print(f"[yellow]⊘[/yellow] {result['reason']}")
        else:
            console.print(f"[green]✓[/green] Compacted {result['source_files']} files ({result['rows']:,} rows)")
            console.print(f"[dim]Output:[/dim] {result['output']}")
            if delete_source:
                console.print(f"[dim]Deleted {result['source_files']} source files[/dim]")

        manager.close()
    except Exception as e:
        console.print(f"[red]✗[/red] Compaction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@table.command("status")
@click.argument("config_file")
def table_status(config_file: str) -> None:
    """Show status of a table."""
    from flowbase.tables.manager import TableManager
    import yaml

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        table_name = config["name"]

        manager = TableManager()
        status = manager.get_status(table_name)

        console.print(f"\n[bold]Table:[/bold] {status['table_name']}")
        console.print(f"\n[bold]Ingestion:[/bold]")
        console.print(f"  Total ingestions: {status['ingestion']['total_ingestions']}")
        console.print(f"  Date range: {status['ingestion']['first_date']} → {status['ingestion']['last_date']}")
        console.print(f"  Total rows: {status['ingestion']['total_rows']:,}")

        console.print(f"\n[bold]Compaction:[/bold]")
        console.print(f"  Total compactions: {status['compaction']['total_compactions']}")
        if status['compaction']['total_compactions'] > 0:
            console.print(f"  Compacted range: {status['compaction']['first_compacted_date']} → {status['compaction']['last_compacted_date']}")

        manager.close()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get status: {e}")
        sys.exit(1)


@table.command("query")
@click.argument("config_file")
@click.argument("sql")
@click.option("--limit", type=int, help="Limit number of rows displayed")
def table_query(config_file: str, sql: str, limit: int) -> None:
    """Query a table using SQL."""
    from flowbase.tables.manager import TableManager
    import yaml

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        table_name = config["name"]

        manager = TableManager()

        with console.status("[bold blue]Executing query..."):
            result = manager.query(table_name, config_file, sql)

        console.print(f"\n[green]✓[/green] {len(result):,} rows returned\n")

        if limit:
            console.print(result.head(limit).to_string())
        else:
            console.print(result.to_string())

        manager.close()
    except Exception as e:
        console.print(f"[red]✗[/red] Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
def scraper() -> None:
    """Manage scrapers (scheduled data collection)."""
    pass


@scraper.command("run")
@click.argument("config_file")
@click.option("--date", help="Date to scrape (YYYY-MM-DD), defaults to today")
def scraper_run(config_file: str, date: str) -> None:
    """Run a scraper and ingest results."""
    from flowbase.scrapers.runner import ScraperRunner

    try:
        runner = ScraperRunner()
        runner.run(config_file, date=date)
    except Exception as e:
        console.print(f"[red]✗[/red] Scraper failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@scraper.command("list")
@click.option("--dir", default="scrapers", help="Directory containing scraper configs")
def scraper_list(dir: str) -> None:
    """List all available scrapers."""
    from flowbase.scrapers.runner import ScraperRunner

    try:
        runner = ScraperRunner()
        scrapers = runner.list_scrapers(dir)

        if not scrapers:
            console.print(f"[yellow]No scrapers found in {dir}/[/yellow]")
            return

        console.print(f"\n[bold]Available scrapers:[/bold]")
        for scraper in scrapers:
            console.print(f"  • {scraper}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list scrapers: {e}")
        sys.exit(1)


@cli.command()
@click.option("--port", default=8501, help="Port to run the UI on")
@click.option("--mode", default="ide", type=click.Choice(["ide", "ml", "explore", "features"]), help="UI mode")
def ui(port: int, mode: str) -> None:
    """Launch the Flowbase UI."""
    import subprocess

    ui_mapping = {
        "ide": ("flowbase/ui/ide.py", "IDE"),
        "ml": ("flowbase/ui/ml_app.py", "ML Studio"),
        "explore": ("flowbase/ui/explorer.py", "Data Explorer"),
        "features": ("flowbase/ui/feature_sets.py", "Feature Sets")
    }

    ui_file, ui_name = ui_mapping[mode]

    console.print(f"[blue]Launching Flowbase {ui_name} on port {port}...[/blue]")
    console.print(f"[dim]Navigate to http://localhost:{port}[/dim]\n")

    try:
        subprocess.run(
            ["streamlit", "run", ui_file, "--server.port", str(port)],
            check=True
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]UI stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to start UI: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
