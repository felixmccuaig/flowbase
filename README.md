# Flowbase

**A declarative ML platform for tabular data that eliminates infrastructure complexity.**

Flowbase lets data scientists build production-ready machine learning pipelines using SQL and YAML configs—no DevOps, no infrastructure setup, no complexity. Just clean data, features, and models.

## Why Flowbase?

Traditional ML platforms force you to choose between:
- **Simple tools** that don't scale (notebooks, pandas scripts)
- **Complex platforms** that require dedicated engineering teams (Kubernetes, Airflow, MLflow)

Flowbase gives you the power of enterprise ML infrastructure with the simplicity of local development:

✅ **SQL-first feature engineering** - All features defined in SQL
✅ **Declarative configs** - YAML-based, version-controlled
✅ **Type-safe data cleaning** - Handle messy data with explicit type casting
✅ **Multiple models, one feature set** - Test different algorithms on the same features
✅ **Local-first development** - Work on your laptop, deploy anywhere
✅ **Built-in comparisons** - Automatic model evaluation and comparison

## Core Concepts

Flowbase follows a clear, linear workflow:

```
Raw Data → Datasets → Features → Models → Evaluations
```

1. **Datasets**: Clean, typed data with quality checks
2. **Features**: SQL-based feature engineering with transformations, aggregations, and window functions
3. **Models**: Train multiple models with different algorithms and feature subsets
4. **Evaluations**: Compare models and select the best performer

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd flowbase

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Your First Pipeline

Let's predict California housing prices:

**1. Define a dataset** (`configs/datasets/housing_clean.yaml`):

```yaml
name: housing_clean
description: Clean California housing data

columns:
  - name: median_income
    type: DOUBLE
  - name: median_house_value
    type: DOUBLE
  - name: ocean_proximity
    type: VARCHAR
    transform: trim

filters:
  - column: median_income
    operator: ">"
    value: 0
  - column: median_house_value
    operator: ">"
    value: 0
```

**2. Compile the dataset**:

```bash
flowbase dataset compile configs/datasets/housing_clean.yaml data/housing.csv
# Output: data/datasets/housing_clean.parquet
```

**3. Define features** (`configs/features/housing_features.yaml`):

```yaml
name: housing_features
dataset: housing_clean

features:
  # Basic ratios
  - name: rooms_per_household
    expression: "total_rooms / households"

  # Income transformations
  - name: income_log
    expression: "LOG(median_income + 1)"

  # One-hot encoding
  - name: ocean_inland
    expression: "CASE WHEN ocean_proximity = 'INLAND' THEN 1 ELSE 0 END"

window_features:
  # Regional averages
  - name: avg_income_by_ocean
    function: AVG
    column: median_income
    partition_by: [ocean_proximity]
```

**4. Compile features**:

```bash
flowbase features compile configs/features/housing_features.yaml \
  -d data/datasets/housing_clean.parquet
# Output: data/features/housing_features.parquet
```

**5. Train models** (`configs/models/random_forest.yaml`):

```yaml
name: housing_random_forest
feature_set: housing_features
target: median_house_value

features:
  - median_income
  - income_log
  - rooms_per_household
  - ocean_inland
  - avg_income_by_ocean

model:
  type: sklearn
  class: ensemble.RandomForestRegressor
  hyperparameters:
    n_estimators: 100
    random_state: 42

split:
  method: random
  test_size: 0.2
  random_state: 42
```

```bash
flowbase model train configs/models/random_forest.yaml \
  -f data/features/housing_features.parquet
# Output: data/models/housing_random_forest.pkl
```

**6. Compare models**:

```bash
flowbase eval compare \
  data/models/model1.pkl \
  data/models/model2.pkl \
  data/models/model3.pkl \
  -n housing_comparison
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Model             ┃ Type    ┃ Test Score┃ RMSE   ┃ MAE    ┃ R²     ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ baseline_simple   │ sklearn │ 0.6139    │ 71,578 │ 51,921 │ 0.6139 │
│ random_forest     │ sklearn │ 0.8167    │ 49,319 │ 31,931 │ 0.8167 │
│ advanced_gbm      │ sklearn │ 0.8411    │ 45,921 │ 29,483 │ 0.8411 │
└───────────────────┴─────────┴───────────┴────────┴────────┴────────┘

Best model: advanced_gbm (test_score: 0.8411)
```

## Examples

Flowbase includes three complete, progressively complex examples that demonstrate the full workflow from raw data to model comparison.

### 1. Iris Classification (`examples/iris/`)

**Perfect for getting started** - A clean, simple dataset to learn the basics.

**The Problem**: Classify iris flowers into 3 species (setosa, versicolor, virginica) based on petal and sepal measurements.

**Dataset**: 150 samples, 4 measurements, 1 target
- `sepal_length`, `sepal_width`, `petal_length`, `petal_width` → `species`

**What You'll Learn**:
- Basic dataset compilation with type casting
- Simple feature engineering (ratios, areas, rankings)
- Training multiple models
- Model comparison for classification

**Run the example**:
```bash
# 1. Clean and type the data
flowbase dataset compile examples/iris/configs/datasets/iris_clean.yaml examples/iris/data/iris.csv

# 2. Engineer features (5 → 19 columns)
#    - Ratios: sepal_ratio, petal_ratio
#    - Areas: sepal_area, petal_area
#    - Rankings: petal_length_rank by species
flowbase features compile examples/iris/configs/features/iris_features.yaml -d data/datasets/iris_clean.parquet

# 3. Train two models
flowbase model train examples/iris/configs/models/random_forest.yaml -f data/features/iris_features.parquet
flowbase model train examples/iris/configs/models/logistic_regression.yaml -f data/features/iris_features.parquet

# 4. Compare results
flowbase eval compare data/models/iris_*.pkl -n iris_comparison
```

**Results**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Model                 ┃ Type    ┃ Test Score ┃ Accuracy ┃ F1     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ iris_random_forest    │ sklearn │ 0.9778     │ 0.9778   │ 0.9776 │
│ iris_logistic_regr... │ sklearn │ 1.0000     │ 1.0000   │ 1.0000 │ ✨
└───────────────────────┴─────────┴────────────┴──────────┴────────┘
```

**Winner**: Logistic Regression with perfect 100% accuracy on test set.

---

### 2. Titanic Survival Prediction (`examples/titanic/`)

**Real-world data messiness** - Learn how to handle missing values, mixed types, and data quality issues.

**The Problem**: Predict passenger survival on the Titanic based on demographics, ticket class, and family relationships.

**Dataset**: 891 passengers, 12 features (messy!)
- **Missing values**: 177 missing ages, 687 missing cabin numbers, 2 missing embarkation ports
- **Type issues**: Mixed case text, numbers stored as strings
- **Complex strings**: Names with titles ("Mr.", "Mrs.", "Master.")

**What You'll Learn**:
- Handling missing values with `allow_null` filters
- Type casting messy data (Age strings → DOUBLE)
- Text normalization (`transform: lower`, `transform: trim`)
- String manipulation (extracting titles from names with `REGEXP_EXTRACT`)
- Creating complex features from multiple columns
- Imputing missing values during feature engineering

**Key Dataset Config Techniques**:
```yaml
columns:
  - name: Age
    type: DOUBLE  # Handles missing values → NULL

  - name: Sex
    type: VARCHAR
    transform: lower  # "Male" → "male"

filters:
  - column: Embarked
    operator: in
    value: ["S", "C", "Q"]
    allow_null: true  # ← Keep rows with missing embarkation
```

**Run the example**:
```bash
# 1. Clean data: handle nulls, normalize text, cast types
flowbase dataset compile examples/titanic/configs/datasets/titanic_clean.yaml examples/titanic/data/titanic.csv

# 2. Engineer 35 features from 12 base columns:
#    - Family features: family_size, is_alone
#    - Age features: age_filled (median imputation), is_child, age_group
#    - Fare features: fare_per_person, fare_category
#    - Text extraction: title from name (Mr, Mrs, Miss, Master, etc.)
#    - Cabin features: has_cabin, cabin_deck
#    - One-hot encoding: sex, class, embarkation port
#    - Window functions: avg_fare_by_class, title_count
flowbase features compile examples/titanic/configs/features/survival_features.yaml -d data/datasets/titanic_clean.parquet

# 3. Train three different models
flowbase model train examples/titanic/configs/models/random_forest.yaml -f data/features/survival_features.parquet
flowbase model train examples/titanic/configs/models/gradient_boosting.yaml -f data/features/survival_features.parquet
flowbase model train examples/titanic/configs/models/logistic_regression.yaml -f data/features/survival_features.parquet

# 4. Compare all models
flowbase eval compare data/models/titanic_*.pkl -n titanic_comparison
```

**Results**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Model                 ┃ Type    ┃ Test Score ┃ Accuracy ┃ F1     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ titanic_random_forest │ sklearn │ 0.8156     │ 0.8156   │ 0.7591 │
│ titanic_gradient_bo...│ sklearn │ 0.8380     │ 0.8380   │ 0.7883 │ ✨
│ titanic_logistic_re...│ sklearn │ 0.8101     │ 0.8101   │ 0.7671 │
└───────────────────────┴─────────┴────────────┴──────────┴────────┘
```

**Winner**: Gradient Boosting with 83.8% accuracy.

**Key Insight**: Even with 20% missing age data and 77% missing cabin data, proper feature engineering (family_size, title extraction, fare_per_person) enables strong predictive performance.

---

### 3. California Housing Price Prediction (`examples/housing/`)

**Production-scale complexity** - The ultimate demonstration of Flowbase's power.

**The Problem**: Predict median house prices for California districts based on location, demographics, and housing characteristics.

**Dataset**: 20,640 districts, 10 base features
- Geographic: `longitude`, `latitude`
- Housing: `housing_median_age`, `total_rooms`, `total_bedrooms` (207 nulls), `households`
- Demographics: `population`
- Economic: `median_income`
- Categorical: `ocean_proximity` (5 categories)
- Target: `median_house_value`

**What You'll Learn**:
- **Scale**: Working with 20K+ rows
- **Comprehensive feature engineering**: 10 base features → 50 engineered features
- **Advanced SQL**: Window functions, aggregations, complex CASE statements
- **Feature strategy**: Building one rich feature set, then selecting different subsets for different models
- **Model comparison**: Testing simple vs. complex approaches on the same data

**The Feature Engineering Strategy**:

We create **50 total features** organized into categories:

**1. Basic Ratios** (4 features)
```yaml
- name: rooms_per_household
  expression: "total_rooms / households"

- name: bedrooms_per_room
  expression: "COALESCE(total_bedrooms, total_rooms * 0.2) / total_rooms"
```

**2. Geographic Features** (7 features)
```yaml
- name: distance_to_sf
  expression: "SQRT(POW(latitude - 37.77, 2) + POW(longitude + 122.42, 2))"

- name: min_distance_to_city
  expression: |
    LEAST(
      SQRT(POW(latitude - 34.05, 2) + POW(longitude + 118.24, 2)),  -- LA
      SQRT(POW(latitude - 37.77, 2) + POW(longitude + 122.42, 2)),  -- SF
      SQRT(POW(latitude - 32.72, 2) + POW(longitude + 117.16, 2))   -- SD
    )
```

**3. Income Transformations** (4 features)
```yaml
- name: income_squared
  expression: "median_income * median_income"

- name: income_log
  expression: "LOG(median_income + 1)"
```

**4. Interaction Features** (2 features)
```yaml
- name: income_age_interaction
  expression: "median_income * housing_median_age"
```

**5. Window Aggregations** (7 features)
```yaml
window_features:
  - name: avg_income_by_ocean
    function: AVG
    column: median_income
    partition_by: [ocean_proximity]

  - name: income_percentile_by_region
    function: PERCENT_RANK
    partition_by: [ocean_proximity]
    order_by: [median_income]
```

**6. One-Hot Encoding** (9 features for ocean_proximity + 4 for income_category)

**The Model Strategy** - Different feature subsets for different approaches:

**Model 1: Baseline Simple** (12 features)
- Just the basics: raw features + simple ratios
- Algorithm: Ridge Regression (linear)
- Philosophy: "Keep it simple, establish a baseline"

**Model 2: Random Forest Medium** (22 features)
- Core features + some engineering, no heavy interactions
- Algorithm: Random Forest (ensemble)
- Philosophy: "Moderate complexity, let trees handle interactions"

**Model 3: Advanced Engineered** (45 features)
- Everything: geographic distances, transformations, interactions, aggregations
- Algorithm: Gradient Boosting (advanced ensemble)
- Philosophy: "Give the model everything we've got"

**Run the example**:
```bash
# 1. Clean and validate data
flowbase dataset compile examples/housing/configs/datasets/housing_clean.yaml examples/housing/data/housing.csv

# 2. Create comprehensive feature set (10 → 50 columns)
flowbase features compile examples/housing/configs/features/housing_features_comprehensive.yaml \
  -d data/datasets/housing_clean.parquet

# 3. Train three models with DIFFERENT feature subsets from the SAME feature file
#    This is the key insight: one feature set, multiple strategies

# Baseline: 12 simple features
flowbase model train examples/housing/configs/models/baseline_simple.yaml \
  -f data/features/housing_features_comprehensive.parquet

# Medium: 22 moderate features
flowbase model train examples/housing/configs/models/random_forest_medium.yaml \
  -f data/features/housing_features_comprehensive.parquet

# Advanced: 45 complex features
flowbase model train examples/housing/configs/models/advanced_engineered.yaml \
  -f data/features/housing_features_comprehensive.parquet

# 4. Compare all three strategies
flowbase eval compare data/models/housing_*.pkl -n housing_comparison
```

**Results**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Model                 ┃ Type    ┃ Test Score ┃ RMSE   ┃ MAE    ┃ R²     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ housing_baseline_si...│ sklearn │ 0.6139     │ 71,578 │ 51,921 │ 0.6139 │
│ housing_random_fore...│ sklearn │ 0.8167     │ 49,319 │ 31,931 │ 0.8167 │
│ housing_advanced_en...│ sklearn │ 0.8411     │ 45,921 │ 29,483 │ 0.8411 │ ✨
└───────────────────────┴─────────┴────────────┴────────┴────────┴────────┘
```

**Winner**: Advanced GBM with R² = 0.841 (84.1% variance explained)

**Key Insights**:

1. **Feature engineering matters**: Going from 12 → 45 features improved R² by **37%** (0.614 → 0.841)

2. **RMSE tells the story**:
   - Baseline: $71,578 average error
   - Advanced: $45,921 average error
   - **$25,657 improvement** in prediction accuracy

3. **One feature set, multiple experiments**: We engineered features once, then tried different combinations. This is real-world ML workflow.

4. **Geographic features are powerful**: Distance to major cities (LA, SF, SD) was highly predictive.

5. **Window aggregations help**: Knowing the average income in your region (ocean_proximity group) provides valuable context.

6. **Model selection via comparison**: Flowbase's automatic comparison made it easy to identify the best approach.

---

## Example Comparison Summary

| Example | Complexity | Samples | Features | Key Learning |
|---------|-----------|---------|----------|--------------|
| **Iris** | Simple | 150 | 5 → 19 | Basic workflow, feature ratios, rankings |
| **Titanic** | Messy | 891 | 12 → 35 | Missing values, type casting, text extraction |
| **Housing** | Production | 20,640 | 10 → 50 | Scale, window functions, feature strategy |

**Recommended Learning Path**:
1. Start with **Iris** to understand the basic workflow
2. Move to **Titanic** to learn data cleaning and handling messiness
3. Master **Housing** to see production-scale feature engineering and model comparison

## Key Features

### 1. SQL-First Feature Engineering

Define features in SQL—the universal language of data:

```yaml
features:
  # Simple expressions
  - name: age_squared
    expression: "age * age"

  # Complex logic
  - name: risk_score
    expression: |
      CASE
        WHEN age < 25 THEN income * 0.5
        WHEN age < 40 THEN income * 1.0
        ELSE income * 1.5
      END

  # Window functions
window_features:
  - name: income_rank
    function: RANK
    partition_by: [state, city]
    order_by: [income DESC]
```

### 2. Type-Safe Data Cleaning

Handle messy real-world data explicitly:

```yaml
columns:
  - name: age
    type: DOUBLE  # Casts VARCHAR "25" → 25.0
    transform: trim

  - name: total_bedrooms
    type: DOUBLE  # Handles NULLs gracefully

  - name: sex
    type: VARCHAR
    transform: lower  # "MALE" → "male"

filters:
  - column: age
    operator: ">"
    value: 0
  - column: ocean_proximity
    operator: in
    value: ["INLAND", "NEAR OCEAN"]
    allow_null: true
```

### 3. Multiple Models, One Feature Set

Test different algorithms and feature combinations:

```yaml
# baseline_simple.yaml - 12 features
features: [longitude, latitude, median_income, rooms_per_household, ...]
model:
  type: sklearn
  class: linear_model.Ridge

# advanced_engineered.yaml - 45 features
features: [longitude, latitude, median_income, income_squared, income_log,
           distance_to_la, distance_to_sf, avg_income_by_ocean, ...]
model:
  type: sklearn
  class: ensemble.GradientBoostingRegressor
```

### 4. Automatic Model Comparison

Built-in evaluation with proper metrics:

- **Classification**: Accuracy, Precision, Recall, F1
- **Regression**: RMSE, MAE, R², MSE

Results saved to JSON for further analysis.

### 5. Model Inference

Once trained, use your models for predictions on new data:

**Example: Iris Classification**
```bash
# Predict iris species from measurements
flowbase model predict iris_simple \
  --input '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}'
```

Output:
```
Making prediction with: iris_simple

Input features:
  sepal length (cm): 5.1
  sepal width (cm): 3.5
  petal length (cm): 1.4
  petal width (cm): 0.2

✓ Prediction complete

Prediction: 0

Class Probabilities:
  Class 0: 0.9766 (97.66%)  ← setosa
  Class 1: 0.0234 (2.34%)   ← versicolor
  Class 2: 0.0000 (0.00%)   ← virginica
```

**Example: Titanic Survival**
```bash
# Predict passenger survival
flowbase model predict titanic_simple \
  --input '{"pclass": 3, "is_male": 1, "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25}'
```

Output:
```
Prediction: 0

Class Probabilities:
  Class 0: 0.8800 (88.00%)  ← died
  Class 1: 0.1200 (12.00%)  ← survived
```

**Example: Housing Price**
```bash
# Predict median house value (in $100k)
flowbase model predict housing_simple \
  --input '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}'
```

Output:
```
Prediction: 4.242422900000001

# This means ~$424k median house value
```

**Input from JSON File**

For complex inputs, use a JSON file:

```json
// input.json
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```

```bash
flowbase model predict iris_simple --input input.json
```

**Key Features**:
- ✅ **Automatic validation**: Checks for missing required features
- ✅ **Correct ordering**: Ensures features are passed in the right order
- ✅ **Probability scores**: Shows class probabilities for classifiers
- ✅ **Type handling**: Automatically handles type conversions
- ✅ **Model metadata**: Uses saved feature names and preprocessing

## CLI Commands

```bash
# Dataset management
flowbase dataset compile <config.yaml> <source.csv> [--output <path>] [--preview]

# Feature engineering
flowbase features compile <config.yaml> --dataset <dataset.parquet> [--output <path>] [--preview]

# Model training
flowbase model train <config.yaml> --features <features.parquet> [--output <models-dir>]

# Model prediction/inference
flowbase model predict <model_name> --input <json-input> [--models-dir <dir>]

# Model evaluation
flowbase eval compare <model1.pkl> <model2.pkl> ... --name <eval-name>
```

## Project Structure

```
flowbase/
├── flowbase/
│   ├── cli/              # Command-line interface
│   ├── core/             # Configuration loading
│   ├── pipelines/        # Dataset & feature compilers
│   ├── models/           # Model training
│   ├── query/            # DuckDB query engine
│   └── storage/          # Storage abstraction
│
├── examples/
│   ├── iris/             # Simple classification
│   ├── titanic/          # Messy data handling
│   └── housing/          # Production-scale example
│
└── data/                 # Generated outputs
    ├── datasets/         # Cleaned, typed data
    ├── features/         # Engineered features
    ├── models/           # Trained models
    └── evals/            # Evaluation results
```

## How It Works

### 1. Dataset Compiler

Transforms raw CSV → clean, typed Parquet:

**Input**: `housing.csv` with mixed types, nulls, outliers
**Config**: Type specifications, transformations, quality filters
**Output**: `housing_clean.parquet` - clean, typed, validated

Generated SQL:
```sql
SELECT
    TRY_CAST(longitude AS DOUBLE) AS longitude,
    TRY_CAST(median_income AS DOUBLE) AS median_income,
    CAST(TRIM(ocean_proximity) AS VARCHAR) AS ocean_proximity
FROM raw_data
WHERE median_income > 0
  AND ocean_proximity IN ('INLAND', 'NEAR OCEAN')
ORDER BY latitude DESC
```

### 2. Feature Compiler

Transforms datasets → feature-rich training data:

**Input**: Clean dataset parquet
**Config**: Feature expressions, window functions
**Output**: Feature-engineered parquet ready for training

Supports:
- **Computed features**: `income * age`, `CASE WHEN ...`
- **Window functions**: `RANK()`, `AVG() OVER (PARTITION BY ...)`
- **One-hot encoding**: Automatic categorical expansion
- **Aggregations**: Group-by operations

### 3. Model Trainer

Trains sklearn/XGBoost/LightGBM models from config:

**Input**: Feature parquet + model config
**Processing**:
- Automatic train/test split
- Missing value imputation
- Model training with hyperparameters
- Metric calculation (classification or regression)

**Output**:
- Trained model (`.pkl`)
- Metadata JSON (features, metrics, hyperparameters)

### 4. Model Evaluator

Compares multiple models with rich tables:

**Input**: Multiple trained model files
**Output**:
- Comparison table (formatted)
- Best model selection
- JSON results file

## Design Principles

1. **SQL is the interface** - All data transformations in SQL
2. **Declarative over imperative** - YAML configs, not code
3. **Type safety matters** - Explicit type casting and validation
4. **Local-first** - Develop on your laptop, deploy anywhere
5. **One feature set, many models** - Reuse features across experiments
6. **Reproducible** - Configs are version-controlled, deterministic

## Roadmap

- [ ] Time-series train/test splits
- [ ] S3/GCS dataset support
- [ ] Trino query engine (for production scale)
- [ ] Model serving API
- [ ] Streaming pipelines
- [ ] Automated hyperparameter tuning
- [ ] Feature store
- [ ] Model monitoring & drift detection

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built for data scientists who want to focus on models, not infrastructure.**
