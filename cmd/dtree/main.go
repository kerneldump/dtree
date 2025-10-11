// Package main implements a small CLI for the dtree library
// providing train, predict, and visualize commands.
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/kerneldump/dtree/dtree"
)

// main dispatches to subcommands: train, predict, visualize.
func main() {
	// Recover from panics to provide a clean error message
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "Fatal error: %v\n", r)
			fmt.Fprintln(os.Stderr, "This is likely a bug. Please report it with the command you ran.")
			os.Exit(2)
		}
	}()

	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	switch cmd {
	case "train":
		trainCmd(args)
	case "predict":
		predictCmd(args)
	case "visualize":
		visualizeCmd(args)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		usage()
		os.Exit(1)
	}
}

// usage prints a short command reference.
func usage() {
	fmt.Println("dtree commands:")
	fmt.Println("  train     --in data.csv --out model.json --label label --format csv")
	fmt.Println("  predict   --in data.csv --model model.json --out preds.jsonl [--csv] [--proba]")
	fmt.Println("  visualize --model model.json --out tree.html [--dot tree.dot]")
}

// trainCmd trains a decision tree from CSV or JSONL and writes a JSON model.
func trainCmd(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	// --in: path to CSV/JSONL; --format: csv|jsonl
	in := fs.String("in", "", "input file (csv or jsonl)")
	out := fs.String("out", "model.json", "output model JSON file")
	format := fs.String("format", "csv", "input format: csv|jsonl")
	// --label: target column name
	label := fs.String("label", "label", "label column name")
	// Optional stopping criteria
	maxDepth := fs.Int("maxDepth", 0, "max depth (0=unlimited)")
	minSamples := fs.Int("minSamples", 0, "min samples per node (0=none)")
	fs.Parse(args)

	if *in == "" {
		fmt.Fprintln(os.Stderr, "--in is required")
		os.Exit(1)
	}
	set, err := readTrainingSet(*in, *format, *label)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read training data: %v\n", err)
		os.Exit(1)
	}
	cfg := dtree.Config{CategoryAttr: *label, Criterion: "entropy", MaxDepth: *maxDepth, MinSamples: *minSamples}
	model, err := dtree.Train(set, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "training failed: %v\n", err)
		os.Exit(1)
	}
	if err := model.SaveJSON(*out); err != nil {
		fmt.Fprintf(os.Stderr, "failed to save model: %v\n", err)
		os.Exit(1)
	}

	// Print success message and model statistics
	fmt.Printf("Model trained successfully and saved to %s\n", *out)
	stats := model.Stats()
	fmt.Printf("Model statistics:\n")
	fmt.Printf("  Tree depth: %d\n", stats.TreeDepth)
	fmt.Printf("  Total nodes: %d\n", stats.TotalNodes)
	fmt.Printf("  Leaf nodes: %d\n", stats.LeafNodes)
	fmt.Printf("  Internal nodes: %d\n", stats.InternalNodes)
	fmt.Printf("  Classes: %d\n", len(stats.Classes))
}

// predictCmd reads data and a JSON model, then outputs predictions.
// By default outputs JSONL; with --csv mirrors input columns plus prediction and optional probabilities.
func predictCmd(args []string) {
	fs := flag.NewFlagSet("predict", flag.ExitOnError)
	// --in/--format: input data; --model: trained model path
	in := fs.String("in", "", "input file (csv or jsonl)")
	modelPath := fs.String("model", "", "model JSON file")
	out := fs.String("out", "", "output file (default stdout)")
	format := fs.String("format", "csv", "input format: csv|jsonl")
	// --csv: output as CSV; --proba: include class probabilities
	asCSV := fs.Bool("csv", false, "output CSV mirroring input")
	proba := fs.Bool("proba", false, "include probabilities in output")
	// --label for CSV header passthrough
	label := fs.String("label", "label", "label column name (for CSV header passthrough)")
	fs.Parse(args)

	if *in == "" || *modelPath == "" {
		fmt.Fprintln(os.Stderr, "--in and --model are required")
		os.Exit(1)
	}
	model, err := dtree.LoadJSON(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load model: %v\n", err)
		os.Exit(1)
	}

	items, headers, err := readItems(*in, *format, *label)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read input data: %v\n", err)
		os.Exit(1)
	}

	var w io.Writer = os.Stdout
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create output file: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		w = f
	}

	if *asCSV {
		cw := csv.NewWriter(w)
		// write header + prediction (and optional proba)
		hdr := append([]string{}, headers...)
		hdr = append(hdr, "prediction")
		if *proba {
			hdr = append(hdr, "proba")
		}
		cw.Write(hdr)
		for i, it := range items {
			pred, err := model.Predict(it)
			if err != nil {
				fmt.Fprintf(os.Stderr, "prediction failed on row %d: %v\n", i+1, err)
				os.Exit(1)
			}
			rec := make([]string, 0, len(headers)+2)
			for _, h := range headers {
				rec = append(rec, fmt.Sprintf("%v", it[h]))
			}
			rec = append(rec, pred)
			if *proba {
				pb, err := model.PredictProba(it)
				if err != nil {
					fmt.Fprintf(os.Stderr, "probability prediction failed on row %d: %v\n", i+1, err)
					os.Exit(1)
				}
				b, _ := json.Marshal(pb)
				rec = append(rec, string(b))
			}
			cw.Write(rec)
		}
		cw.Flush()
		if err := cw.Error(); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write CSV output: %v\n", err)
			os.Exit(1)
		}
		if *out != "" {
			fmt.Printf("Predictions written to %s\n", *out)
		}
		return
	}

	// JSONL output
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i, it := range items {
		pred, err := model.Predict(it)
		if err != nil {
			fmt.Fprintf(os.Stderr, "prediction failed on row %d: %v\n", i+1, err)
			os.Exit(1)
		}
		out := map[string]interface{}{"input": it, "prediction": pred}
		if *proba {
			pb, err := model.PredictProba(it)
			if err != nil {
				fmt.Fprintf(os.Stderr, "probability prediction failed on row %d: %v\n", i+1, err)
				os.Exit(1)
			}
			out["proba"] = pb
		}
		if err := enc.Encode(out); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write JSONL output: %v\n", err)
			os.Exit(1)
		}
	}
	bw.Flush()
	if *out != "" {
		fmt.Printf("Predictions written to %s\n", *out)
	}
}

// visualizeCmd renders the model to HTML, and optionally Graphviz DOT.
func visualizeCmd(args []string) {
	fs := flag.NewFlagSet("visualize", flag.ExitOnError)
	modelPath := fs.String("model", "", "model JSON file")
	outHTML := fs.String("out", "tree.html", "output HTML file")
	outDOT := fs.String("dot", "", "optional DOT output file")
	fs.Parse(args)

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(1)
	}
	model, err := dtree.LoadJSON(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load model: %v\n", err)
		os.Exit(1)
	}
	if err := model.ToHTML(*outHTML); err != nil {
		fmt.Fprintf(os.Stderr, "failed to write HTML: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("HTML visualization written to %s\n", *outHTML)

	if *outDOT != "" {
		if err := os.WriteFile(*outDOT, []byte(model.ToDOT()), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write DOT file: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("DOT file written to %s\n", *outDOT)
	}
}

// IO helpers

// readTrainingSet loads and validates a dataset for training.
func readTrainingSet(path, format, label string) (dtree.TrainingSet, error) {
	items, _, err := readItems(path, format, label)
	if err != nil {
		return nil, err
	}
	// ensure label exists
	for i, it := range items {
		if _, ok := it[label]; !ok {
			return nil, fmt.Errorf("missing label '%s' in row %d", label, i+1)
		}
	}
	return dtree.TrainingSet(items), nil
}

// readItems loads rows from CSV (using header) or JSONL.
// Returns a slice of items and the header order (for CSV output mirroring).
func readItems(path, format, label string) ([]dtree.TrainingItem, []string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer f.Close()
	switch strings.ToLower(format) {
	case "csv":
		r := csv.NewReader(f)
		r.TrimLeadingSpace = true
		header, err := r.Read()
		if err != nil {
			return nil, nil, fmt.Errorf("cannot read CSV header: %w", err)
		}
		var items []dtree.TrainingItem
		rowNum := 2 // Start at 2 (1 is header)
		for {
			rec, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				return nil, nil, fmt.Errorf("error reading CSV row %d: %w", rowNum, err)
			}
			if len(rec) != len(header) {
				return nil, nil, fmt.Errorf("row %d has %d columns but header has %d", rowNum, len(rec), len(header))
			}
			it := dtree.TrainingItem{}
			for i, h := range header {
				it[h] = parseCSVValue(rec[i])
			}
			items = append(items, it)
			rowNum++
		}
		if len(items) == 0 {
			return nil, nil, fmt.Errorf("CSV file is empty (no data rows)")
		}
		return items, header, nil
	case "jsonl":
		var items []dtree.TrainingItem
		sc := bufio.NewScanner(f)
		lineNum := 1
		for sc.Scan() {
			var m map[string]interface{}
			if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
				return nil, nil, fmt.Errorf("invalid JSON on line %d: %w", lineNum, err)
			}
			items = append(items, dtree.TrainingItem(m))
			lineNum++
		}
		if err := sc.Err(); err != nil {
			return nil, nil, fmt.Errorf("error reading JSONL: %w", err)
		}
		if len(items) == 0 {
			return nil, nil, fmt.Errorf("JSONL file is empty")
		}
		// collect headers from first item (best-effort)
		hdr := []string{}
		if len(items) > 0 {
			for k := range items[0] {
				hdr = append(hdr, k)
			}
		}
		return items, hdr, nil
	default:
		return nil, nil, fmt.Errorf("unknown format: %s (must be 'csv' or 'jsonl')", format)
	}
}

// parseCSVValue converts CSV cell strings to float64, bool, or leaves as string.
func parseCSVValue(s string) interface{} {
	if s == "" {
		return nil
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	if s == "true" {
		return true
	}
	if s == "false" {
		return false
	}
	return s
}
