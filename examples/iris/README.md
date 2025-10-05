# Iris Classification Example

This example demonstrates the complete Flowbase workflow using the classic Iris dataset.

## Project Structure

```
iris/
├── data/
│   └── iris.csv                          # Raw data
├── configs/
│   ├── sources/
│   │   └── iris_raw.yaml                 # Data source metadata
│   ├── datasets/
│   │   └── iris_clean.yaml               # Dataset with types & quality checks
│   ├── features/
│   │   └── iris_features.yaml            # Feature engineering
│   ├── models/
│   │   ├── random_forest.yaml            # Random Forest model config
│   │   └── logistic_regression.yaml      # Logistic Regression config
│   └── evals/
│       └── compare_models.yaml           # Model comparison config
└── README.md
```

## Workflow

### 1. Data Source

**Purpose**: Document raw data files

```yaml
# configs/sources/iris_raw.yaml
name: iris_raw
type: csv
path: examples/iris/data/iris.csv
```

This is mainly for documentation - tells you where the raw data lives and what it looks like.

### 2. Dataset (Clean & Type)

**Purpose**: Clean raw data and define proper types

```bash
flowbase dataset compile \
  examples/iris/configs/datasets/iris_clean.yaml \
  examples/iris/data/iris.csv \
  --preview
```

**What it does:**
- Casts columns to proper types (DOUBLE, VARCHAR, etc.)
- Applies data quality filters
- Removes invalid rows
- Saves to `data/datasets/iris_clean.parquet`

**Key config:**
```yaml
columns:
  - name: sepal_length
    type: DOUBLE
  - name: species
    type: VARCHAR
    transform: trim

filters:
  - column: sepal_length
    operator: ">"
    value: 0
```

### 3. Feature Set (Engineer Features)

**Purpose**: Create ML features from clean dataset

```bash
flowbase features compile \
  examples/iris/configs/features/iris_features.yaml \
  --dataset data/datasets/iris_clean.parquet \
  --preview
```

**What it does:**
- References the clean dataset (no type errors!)
- Creates computed features (ratios, areas, etc.)
- Applies window functions (rankings, aggregations)
- Saves to `data/features/iris_features.parquet`

**Key features:**
```yaml
features:
  - name: sepal_ratio
    expression: "sepal_length / sepal_width"

  - name: petal_area
    expression: "3.14159 * (petal_length / 2) * (petal_width / 2)"

window_features:
  - name: petal_length_rank
    function: RANK
    partition_by: [species]
    order_by: [petal_length DESC]
```

### 4. Model (Train)

**Purpose**: Define model architecture and hyperparameters

```bash
# Coming soon - model training via CLI
flowbase model train examples/iris/configs/models/random_forest.yaml
```

**Key config:**
```yaml
feature_set: iris_features
target: species
features:
  - sepal_length
  - petal_length
  - sepal_ratio
  - petal_area

model:
  type: sklearn
  class: ensemble.RandomForestClassifier
  hyperparameters:
    n_estimators: 100
    max_depth: 10

split:
  method: random
  test_size: 0.3
```

### 5. Evaluation (Compare Models)

**Purpose**: Compare multiple models on same data

```bash
# Coming soon - model evaluation
flowbase eval run examples/iris/configs/evals/compare_models.yaml
```

**What it does:**
- Trains multiple models (Random Forest, Logistic Regression)
- Evaluates on same test set
- Generates comparison metrics
- Creates confusion matrices, ROC curves
- Saves results to `examples/iris/results/`

## Quick Start

```bash
# 1. Navigate to flowbase directory
cd /Users/felixmccuaig/flowbase

# 2. Activate environment
source venv/bin/activate

# 3. Compile dataset
flowbase dataset compile \
  examples/iris/configs/datasets/iris_clean.yaml \
  examples/iris/data/iris.csv

# 4. Compile features
flowbase features compile \
  examples/iris/configs/features/iris_features.yaml \
  --dataset data/datasets/iris_clean.parquet

# 5. Train models (coming soon)
# flowbase model train examples/iris/configs/models/random_forest.yaml

# 6. Evaluate (coming soon)
# flowbase eval run examples/iris/configs/evals/compare_models.yaml
```

## Benefits of This Approach

1. **Declarative**: Everything is defined in YAML configs
2. **Version controlled**: All configs can be git-tracked
3. **Reproducible**: Same configs = same results
4. **Type safe**: Dataset step ensures correct types
5. **Modular**: Swap datasets, features, models independently
6. **No code**: Data scientists don't need to write Python

## Next Steps

- Add your own features in `iris_features.yaml`
- Try different models (XGBoost, LightGBM)
- Experiment with different train/test splits
- Add more data quality rules
