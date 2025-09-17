# dtree

Decision tree (classification) library and CLI in Go.

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
```
import "github.com/kerneldump/dtree/dtree"

set := dtree.TrainingSet{
  dtree.TrainingItem{"x": 1.0, "y": "a", "label": "A"},
}
cfg := dtree.Config{CategoryAttr: "label"}
model := dtree.Train(set, cfg)
pred := model.Predict(dtree.TrainingItem{"x": 2.0, "y": "a"})
```

Examples
- PlayTennis: `examples/playtennis.csv`, `examples/playtennis.jsonl`

Project Flow
- See `flow.svg` for a high-level diagram of how the CLI and library interact (train/predict/visualize and file I/O).
- Open directly (e.g., `open flow.svg`) or view it in your editor.

Makefile
- `make build`, `make test`, `make example-all`, `make clean`

License
- Apache-2.0
