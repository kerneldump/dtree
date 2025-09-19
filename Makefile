GO ?= go
BIN ?= bin/dtree

.PHONY: build install test fmt clean
.PHONY: example-playtennis example-customer demo

build:
	@mkdir -p bin
	$(GO) build -o $(BIN) ./cmd/dtree

install:
	$(GO) install ./cmd/dtree

test:
	$(GO) test ./...

fmt:
	$(GO) fmt ./...

example-playtennis: build
	@mkdir -p models predictions visualizations
	$(BIN) train --in examples/playtennis.csv --format csv --label Play --out models/playtennis.json
	$(BIN) predict --in examples/playtennis.csv --format csv --label Play --model models/playtennis.json --csv --proba --out predictions/playtennis_preds.csv
	$(BIN) visualize --model models/playtennis.json --out visualizations/playtennis.html --dot visualizations/playtennis.dot
	@echo "PlayTennis example complete. View: visualizations/playtennis.html"

example-customer: build
	@mkdir -p models predictions visualizations
	$(BIN) train --in examples/customer_segmentation.csv --format csv --label Segment --out models/customer.json --maxDepth 8 --minSamples 1
	$(BIN) predict --in examples/customer_segmentation.csv --format csv --label Segment --model models/customer.json --csv --proba --out predictions/customer_preds.csv
	$(BIN) visualize --model models/customer.json --out visualizations/customer.html --dot visualizations/customer.dot
	@echo "Customer Segmentation example complete. View: visualizations/customer.html"

demo: example-playtennis example-customer
	@echo ""
	@echo "Examples complete:"
	@echo "  Simple: visualizations/playtennis.html"
	@echo "  Complex: visualizations/customer.html"

clean:
	rm -f model.json preds.csv tree.html tree.dot
	rm -rf bin models predictions visualizations
	@echo "Cleaned up generated files"

# Legacy compatibility  
example-train: example-playtennis
example-predict: 
	$(BIN) predict --in examples/playtennis.csv --format csv --label Play --model models/playtennis.json --csv --proba --out preds.csv
example-visualize:
	$(BIN) visualize --model models/playtennis.json --out tree.html --dot tree.dot
example-all: demo