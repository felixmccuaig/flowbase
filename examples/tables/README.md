# Table Management: Incremental Loading & Compaction

Flowbase tables provide a system for managing incremental data loading with automatic compaction to avoid expensive S3 LIST operations.

## Concept

A **table** is a folder containing partitioned data files (daily) that get automatically compacted into larger files (monthly) to optimize query performance.

```
data/tables/sales/
├── 2024-01-01.parquet  # Daily files (deleted after compaction)
├── 2024-01-02.parquet
├── ...
├── 2024-01-31.parquet
└── 2024-01.parquet     # Monthly compacted file (all of January)
```

## Benefits

1. **Incremental loading**: Ingest data day by day without reprocessing historical data
2. **Efficient queries**: DuckDB reads fewer, larger files instead of many small ones
3. **Cost savings**: Avoid expensive S3 LIST operations in cloud deployments
4. **Metadata tracking**: SQLite database tracks what's been ingested and compacted
5. **Data cleaning**: Apply dataset cleaning rules during ingestion

## Quick Start

### 1. Create a Table Configuration

`examples/tables/sales/configs/sales_table.yaml`:

```yaml
name: sales
description: Daily sales transactions table

storage:
  type: local
  base_path: data/tables/sales/
  format: parquet

partitioning:
  strategy: flat  # Flat file structure
  pattern: "{date}.parquet"  # e.g., 2024-01-15.parquet
  date_format: "%Y-%m-%d"

compaction:
  enabled: true

  trigger:
    type: manual  # or schedule: "0 3 1 * *" (cron)
    delay_days: 7  # Wait 7 days after month ends

  strategy:
    type: monthly  # Daily files → monthly file
    output_pattern: "{year}-{month}.parquet"
    delete_source: false  # Keep daily files or delete?
    lookback_months: 1

# Optional: Apply dataset cleaning during ingestion
ingestion:
  dataset_config: examples/tables/sales/configs/sales_dataset.yaml
```

### 2. Create the Table

```bash
flowbase table create examples/tables/sales/configs/sales_table.yaml
```

Output:
```
✓ Table created: sales
Base path: data/tables/sales/
```

### 3. Ingest Daily Data

```bash
# Ingest a single day
flowbase table ingest examples/tables/sales/configs/sales_table.yaml \
  examples/tables/sales/data/sales_2024-01-01.csv \
  --date 2024-01-01 \
  --dataset-config examples/tables/sales/configs/sales_dataset.yaml
```

Output:
```
✓ Ingested 88 rows
Destination: data/tables/sales/2024-01-01.parquet
```

**Ingest multiple days** (bash loop):
```bash
for day in {01..31}; do
  flowbase table ingest examples/tables/sales/configs/sales_table.yaml \
    examples/tables/sales/data/sales_2024-01-$day.csv \
    --date 2024-01-$day \
    --dataset-config examples/tables/sales/configs/sales_dataset.yaml
done
```

### 4. Check Table Status

```bash
flowbase table status examples/tables/sales/configs/sales_table.yaml
```

Output:
```
Table: sales

Ingestion:
  Total ingestions: 31
  Date range: 2024-01-01 → 2024-01-31
  Total rows: 2,239

Compaction:
  Total compactions: 0
```

### 5. Query the Table

```bash
flowbase table query examples/tables/sales/configs/sales_table.yaml \
  "SELECT date, COUNT(*) as sales_count, SUM(quantity * price) as revenue
   FROM sales
   GROUP BY date
   ORDER BY date" \
  --limit 10
```

Output:
```
✓ 10 rows returned

        date  sales_count    revenue
0 2024-01-01           88  109076.01
1 2024-01-02           85  119808.98
2 2024-01-03           60   91806.56
...
```

**Note**: DuckDB automatically reads all `.parquet` files in the table directory.

### 6. Compact Monthly Data

```bash
# Compact January (but keep daily files)
flowbase table compact examples/tables/sales/configs/sales_table.yaml \
  --period 2024-01

# Or delete daily files after compaction
flowbase table compact examples/tables/sales/configs/sales_table.yaml \
  --period 2024-01 \
  --delete-source
```

Output:
```
✓ Compacted 31 files (2,239 rows)
Output: data/tables/sales/2024-01.parquet
Deleted 31 source files
```

After compaction:
```
data/tables/sales/
└── 2024-01.parquet  # Single monthly file (31 daily files → 1 monthly file)
```

### 7. Verify Compaction

```bash
flowbase table status examples/tables/sales/configs/sales_table.yaml
```

Output:
```
Table: sales

Ingestion:
  Total ingestions: 31
  Date range: 2024-01-01 → 2024-01-31
  Total rows: 2,239

Compaction:
  Total compactions: 1
  Compacted range: 2024-01-01 → 2024-01-31
```

Query still works seamlessly:
```bash
flowbase table query examples/tables/sales/configs/sales_table.yaml \
  "SELECT COUNT(*) as total_sales FROM sales"
```

Output:
```
✓ 1 rows returned

   total_sales
0         2239
```

## Configuration Options

### Storage

```yaml
storage:
  type: local  # or s3 (future)
  base_path: data/tables/sales/
  format: parquet  # or csv
```

### Partitioning

```yaml
partitioning:
  strategy: flat  # Flat file structure (2024-01-01.parquet)
  pattern: "{date}.parquet"  # Filename pattern
  date_format: "%Y-%m-%d"  # Python datetime format
```

For hierarchical partitioning:
```yaml
partitioning:
  strategy: hive  # Hive-style partitioning
  pattern: "year={year}/month={month}/day={day}/data.parquet"
```

### Compaction

```yaml
compaction:
  enabled: true

  trigger:
    type: manual  # manual or schedule
    schedule: "0 3 1 * *"  # Cron expression (if schedule type)
    delay_days: 7  # Wait N days after period ends before compacting

  strategy:
    type: monthly  # daily → monthly
    output_pattern: "{year}-{month}.parquet"
    delete_source: false  # Delete daily files after compaction?
    lookback_months: 1  # Only compact recent months
```

### Dataset Cleaning

Apply cleaning rules during ingestion:

```yaml
ingestion:
  dataset_config: configs/dataset_clean.yaml
```

Example `dataset_clean.yaml`:
```yaml
name: sales
columns:
  - name: date
    type: DATE
  - name: quantity
    type: INTEGER
  - name: price
    type: DOUBLE
  - name: region
    type: VARCHAR
    transform: upper

filters:
  - column: quantity
    operator: ">"
    value: 0
  - column: price
    operator: ">"
    value: 0
```

## CLI Commands

```bash
# Create table
flowbase table create <config.yaml>

# Ingest data for a date
flowbase table ingest <config.yaml> <source-file> --date YYYY-MM-DD [--dataset-config <config>]

# Check status
flowbase table status <config.yaml>

# Query table
flowbase table query <config.yaml> "<SQL>" [--limit N]

# Compact period
flowbase table compact <config.yaml> --period YYYY-MM [--delete-source]
```

## Metadata Tracking

Flowbase tracks all operations in a SQLite database: `data/tables/.metadata.db`

**Tables**:
- `ingestion_log` - Records each daily ingestion
- `compaction_log` - Records each compaction operation
- `table_registry` - Registry of all tables

You can query the metadata directly:
```bash
sqlite3 data/tables/.metadata.db "SELECT * FROM ingestion_log WHERE table_name='sales'"
```

## Use Cases

### 1. Daily Sales Data Pipeline

Ingest daily sales every morning at 2 AM, compact monthly:

```yaml
compaction:
  trigger:
    type: schedule
    schedule: "0 3 1 * *"  # 3 AM on 1st of each month
    delay_days: 7  # Wait 7 days into new month before compacting
```

### 2. Event Streaming Data

Ingest event data continuously, compact weekly:

```yaml
compaction:
  strategy:
    type: weekly  # Daily files → weekly files
    output_pattern: "{year}-W{week}.parquet"
```

### 3. Historical Data with Recent Detail

Keep daily files for recent data, monthly for history:

```yaml
compaction:
  trigger:
    delay_days: 30  # Only compact data older than 30 days
  strategy:
    delete_source: true  # Delete daily files after compaction
```

## Performance Benefits

**Before Compaction** (31 daily files):
- DuckDB must LIST all files: 31 LIST operations
- Read 31 small files: Higher overhead
- S3 LIST costs: $0.005 per 1,000 requests

**After Compaction** (1 monthly file):
- DuckDB LISTs 1 file: 1 LIST operation
- Read 1 large file: Lower overhead
- 97% reduction in LIST operations

For 1 year of daily data:
- Without compaction: 365 files → 365 LIST ops
- With monthly compaction: 12 files → 12 LIST ops
- **97% reduction in LIST operations**

## Next Steps

1. Run the sales example: `cd examples/tables/sales/`
2. Explore the configs: `configs/sales_table.yaml`
3. Test ingestion and compaction
4. Build your own table for your data!

---

**Key Insight**: Tables give you the performance of a data warehouse without the complexity. Ingest incrementally, compact automatically, query efficiently.
