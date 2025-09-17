GO ?= go
BIN ?= bin/dtree

.PHONY: build install test fmt clean example-train example-predict example-visualize example-all

build:
	@mkdir -p bin
	$(GO) build -o $(BIN) ./cmd/dtree

install:
	$(GO) install ./cmd/dtree

test:
	$(GO) test ./...

fmt:
	$(GO) fmt ./...

example-train: build
	$(BIN) train --in examples/playtennis.csv --format csv --label Play --out model.json

example-predict: build
	$(BIN) predict --in examples/playtennis.csv --format csv --label Play --model model.json --csv --proba --out preds.csv

example-visualize: build
	$(BIN) visualize --model model.json --out tree.html --dot tree.dot

example-all: example-train example-predict example-visualize

clean:
	rm -f model.json preds.csv tree.html tree.dot
	rm -rf bin

