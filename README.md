# dtree

A decision tree classification library and CLI tool in Go.

## What is a Decision Tree?

Decision trees are a fundamental machine learning algorithm used for classification tasks. They work by partitioning data into subsets based on attribute values, creating a tree-like model of decisions. Each internal node tests an attribute, each branch represents a test outcome, and each leaf node represents a class prediction.

This implementation uses entropy-based information gain to find optimal splits, automatically handles numeric and categorical features, and gracefully manages missing values.

## Features

- Entropy-based splitting with automatic feature type detection
- Missing value handling (routes to larger child branch)
- JSON model serialization
- Interactive HTML visualization
- Graphviz DOT export
- CSV and JSONL input support
- Batch predictions with probability scores

## Installation

### Install CLI tool
```bash
go install github.com/kerneldump/dtree/cmd/dtree@latest
```

### Use as library
```bash
go get github.com/kerneldump/dtree/dtree
```

### Build from source
```bash
git clone https://github.com/kerneldump/dtree.git
cd dtree
make build
```

## Quick Start (CLI)

### 1. Train a model
```bash
dtree train --in data.csv --label target_column --out model.json
```

### 2. Make predictions
```bash
# Output as CSV with probabilities
dtree predict --in data.csv --model model.json --csv --proba --out predictions.csv

# Output as JSONL
dtree predict --in data.csv --model model.json --out predictions.jsonl
```

### 3. Visualize the tree
```bash
dtree visualize --model model.json --out tree.html
```

### 4. Try the examples
```bash
make demo
open visualizations/playtennis.html    # Simple example
open visualizations/customer.html      # Complex example
```

## CLI Usage

### Training
```bash
dtree train \
  --in examples/playtennis.csv \
  --format csv \
  --label Play \
  --out model.json \
  --maxDepth 10 \
  --minSamples 5
```

**Flags:**
- `--in`: Input file path (required)
- `--format`: Input format: `csv` or `jsonl` (default: `csv`)
- `--label`: Target column name (default: `label`)
- `--out`: Output model file (default: `model.json`)
- `--maxDepth`: Maximum tree depth, 0 for unlimited (default: `0`)
- `--minSamples`: Minimum samples per node, 0 for no limit (default: `0`)

### Prediction
```bash
dtree predict \
  --in test_data.csv \
  --model model.json \
  --csv \
  --proba \
  --out predictions.csv
```

**Flags:**
- `--in`: Input file path (required)
- `--model`: Trained model file (required)
- `--format`: Input format: `csv` or `jsonl` (default: `csv`)
- `--label`: Label column name for CSV header passthrough (default: `label`)
- `--out`: Output file, uses stdout if not specified
- `--csv`: Output as CSV mirroring input columns
- `--proba`: Include class probabilities in output

### Visualization
```bash
dtree visualize \
  --model model.json \
  --out tree.html \
  --dot tree.dot
```

**Flags:**
- `--model`: Trained model file (required)
- `--out`: Output HTML file (default: `tree.html`)
- `--dot`: Optional DOT file for Graphviz

## Go Library Usage

### Basic Example

```go
package main

import (
    "fmt"
    "log"
    "github.com/kerneldump/dtree/dtree"
)

func main() {
    // Prepare training data
    data := dtree.TrainingSet{
        dtree.TrainingItem{"outlook": "sunny", "temp": 85.0, "humidity": 85.0, "play": "no"},
        dtree.TrainingItem{"outlook": "sunny", "temp": 80.0, "humidity": 90.0, "play": "no"},
        dtree.TrainingItem{"outlook": "overcast", "temp": 83.0, "humidity": 86.0, "play": "yes"},
        dtree.TrainingItem{"outlook": "rain", "temp": 70.0, "humidity": 96.0, "play": "yes"},
        dtree.TrainingItem{"outlook": "rain", "temp": 68.0, "humidity": 80.0, "play": "yes"},
    }
    
    // Configure and train
    config := dtree.Config{
        CategoryAttr: "play",
        MaxDepth:     10,
        MinSamples:   2,
    }
    
    model, err := dtree.Train(data, config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Make a prediction
    testItem := dtree.TrainingItem{
        "outlook":  "sunny",
        "temp":     75.0,
        "humidity": 70.0,
    }
    
    prediction, err := model.Predict(testItem)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Prediction:", prediction)
    
    // Get probabilities
    probabilities, err := model.PredictProba(testItem)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Probabilities:", probabilities)
    
    // Save model
    if err := model.SaveJSON("model.json"); err != nil {
        log.Fatal(err)
    }
    
    // Load model
    loadedModel, err := dtree.LoadJSON("model.json")
    if err != nil {
        log.Fatal(err)
    }
    
    // Get model statistics
    stats := loadedModel.Stats()
    fmt.Printf("Tree depth: %d, Total nodes: %d, Leaf nodes: %d\n",
        stats.TreeDepth, stats.TotalNodes, stats.LeafNodes)
}
```

### Advanced Configuration

```go
config := dtree.Config{
    CategoryAttr:      "label",           // Required: target column
    IgnoredAttributes: []string{"id"},    // Optional: columns to ignore
    Criterion:         "entropy",         // Splitting criterion (currently only entropy)
    MaxDepth:          15,                // Optional: limit tree depth (0 = unlimited)
    MinSamples:        10,                // Optional: min samples to split (0 = no limit)
}
```

### Batch Predictions

```go
// Predict multiple items at once
items := []dtree.TrainingItem{
    {"feature1": "value1", "feature2": 42.0},
    {"feature1": "value2", "feature2": 38.0},
}

predictions, err := model.PredictBatch(items)
probabilities, err := model.PredictProbaBatch(items)
```

## Data Format

### CSV Format
```csv
Outlook,Temperature,Humidity,Wind,Play
sunny,85,85,false,no
overcast,83,86,false,yes
rain,70,96,false,yes
```

- First row must be headers
- Numeric values are auto-detected
- Boolean values: `true`/`false`
- Missing values are handled automatically

### JSONL Format
```jsonl
{"outlook": "sunny", "temperature": 85, "humidity": 85, "wind": false, "play": "no"}
{"outlook": "overcast", "temperature": 83, "humidity": 86, "wind": false, "play": "yes"}
{"outlook": "rain", "temperature": 70, "humidity": 96, "wind": false, "play": "yes"}
```

- One JSON object per line
- Attribute names can vary between records
- Values can be strings, numbers, or booleans

## Examples

### PlayTennis Dataset
Classic machine learning example predicting whether to play tennis based on weather conditions.

```bash
make example-playtennis
open visualizations/playtennis.html
```

**Files:**
- `examples/playtennis.csv` - Training data
- `models/playtennis.json` - Trained model
- `predictions/playtennis_preds.csv` - Predictions with probabilities
- `visualizations/playtennis.html` - Interactive tree visualization

### Customer Segmentation Dataset
More complex example with multiple numeric and categorical features.

```bash
make example-customer
open visualizations/customer.html
```

**Files:**
- `examples/customer_segmentation.csv` - Training data
- `models/customer.json` - Trained model
- `predictions/customer_preds.csv` - Predictions
- `visualizations/customer.html` - Interactive tree visualization

## Model Format

Models are saved as JSON with the following structure:

```json
{
  "root": {
    "attribute": "outlook",
    "predicateName": "==",
    "pivot": "overcast",
    "match": { ... },
    "noMatch": { ... },
    "classCounts": {"yes": 5, "no": 3}
  },
  "config": {
    "categoryAttr": "play",
    "criterion": "entropy",
    "maxDepth": 10,
    "minSamples": 5
  }
}
```

## Makefile Commands

```bash
make build              # Build CLI binary
make install            # Install CLI to $GOPATH/bin
make test               # Run tests
make fmt                # Format code
make clean              # Remove generated files
make demo               # Run all examples
make example-playtennis # Run PlayTennis example
make example-customer   # Run Customer Segmentation example
```

## Requirements

- Go 1.22 or higher

## Algorithm Details

- **Splitting Criterion:** Information gain using Shannon entropy
- **Feature Types:** Automatically detects numeric (>=) vs categorical (==) features
- **Missing Values:** Routes to the child with more training samples
- **Stopping Criteria:** Pure node, max depth reached, or min samples threshold
- **Prediction:** Traverses tree; falls back to majority class if path is blocked

## Limitations

- Classification only (no regression)
- Only entropy criterion (no Gini impurity)
- No pruning (may overfit on noisy data)
- Single tree only (no ensemble methods like Random Forest)

## Use Cases

**Good for:**
- Interpretable models where you need to explain decisions
- Datasets with mixed numeric and categorical features
- Problems where feature interactions matter
- Educational purposes and prototyping

**Not ideal for:**
- Very large datasets (consider Random Forests or Gradient Boosting)
- High-dimensional data with weak signals
- When you need the absolute best accuracy (use ensembles)

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
