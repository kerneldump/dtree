# dtree

Decision tree (classification) library and CLI in Go.

## Decision Trees in Machine Learning

Decision trees are a fundamental algorithm in supervised machine learning, used for both classification and regression tasks. They work by partitioning a dataset into smaller and smaller subsets based on the values of its attributes, creating a tree-like model of decisions. Each internal node in the tree represents a "test" on an attribute (e.g., whether a coin flip is heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (a decision taken after computing all attributes). The paths from root to leaf represent classification rules.

This project implements a classification decision tree, a type of decision tree that aims to predict a categorical target variable. It uses entropy, a measure of impurity or disorder, to find the best splits in the data. By recursively splitting the data, the tree learns a set of rules that can be used to classify new, unseen data.

Features
- Entropy-based splits, auto-detect numeric vs categorical features
- Missing values handled (routes to larger child); majority fallback at leaves
- JSON model save/load
- HTML flow diagram rendering and Graphviz DOT export
- CLI for training, predicting, and visualizing

Requirements
- Go 1.22+

Install
- Library: `go get github.com/kerneldump/dtree/dtree`
- CLI: `go build -o bin/dtree ./cmd/dtree` or `make build`

Quickstart (CLI)
- Train:
  - `bin/dtree train --in examples/playtennis.csv --format csv --label Play --out model.json`
- Predict:
  - JSONL: `bin/dtree predict --in examples/playtennis.csv --format csv --label Play --model model.json --out preds.jsonl`
  - CSV + proba: `bin/dtree predict --in examples/playtennis.csv --format csv --label Play --model model.json --csv --proba --out preds.csv`
- Visualize:
  - `bin/dtree visualize --model model.json --out tree.html --dot tree.dot`

Library (Go)
```go
import (
  "log"
  "github.com/kerneldump/dtree/dtree"
)

func main() {
    set := dtree.TrainingSet{
        dtree.TrainingItem{"x": 1.0, "y": "a", "label": "A"},
    }
    cfg := dtree.Config{CategoryAttr: "label"}
    model, err := dtree.Train(set, cfg)
    if err != nil {
        log.Fatal(err)
    }
    pred, err := model.Predict(dtree.TrainingItem{"x": 2.0, "y": "a"})
    if err != nil {
        log.Fatal(err)
    }
}
```

## Examples

PlayTennis: examples/playtennis.csv, examples/playtennis.jsonl

## Project Flow

See flow.svg for a high-level diagram of how the CLI and library interact (train/predict/visualize and file I/O).
Open directly (e.g., open flow.svg) or view it in your editor.

## Makefile

make build, make test, make example-all, make clean

## License

Apache-2.0