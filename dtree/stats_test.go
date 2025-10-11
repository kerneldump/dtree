package dtree

import (
	"testing"
)

func TestStats_SimpleTree(t *testing.T) {
	// Create a simple tree: root -> 2 leaves
	ts := TrainingSet{
		TrainingItem{"x": 1.0, "label": "A"},
		TrainingItem{"x": 2.0, "label": "B"},
	}
	cfg := Config{CategoryAttr: "label"}
	model, err := Train(ts, cfg)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	stats := model.Stats()

	// Should have 3 nodes: 1 root + 2 leaves
	if stats.TotalNodes != 3 {
		t.Errorf("expected 3 total nodes, got %d", stats.TotalNodes)
	}

	if stats.LeafNodes != 2 {
		t.Errorf("expected 2 leaf nodes, got %d", stats.LeafNodes)
	}

	if stats.InternalNodes != 1 {
		t.Errorf("expected 1 internal node, got %d", stats.InternalNodes)
	}

	if stats.TreeDepth != 1 {
		t.Errorf("expected depth 1, got %d", stats.TreeDepth)
	}

	if len(stats.Classes) != 2 {
		t.Errorf("expected 2 classes, got %d", len(stats.Classes))
	}
}

func TestStats_PlayTennis(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"Outlook": "sunny", "Temperature": 85.0, "Humidity": 85.0, "Wind": false, "Play": "no"},
		TrainingItem{"Outlook": "sunny", "Temperature": 80.0, "Humidity": 90.0, "Wind": true, "Play": "no"},
		TrainingItem{"Outlook": "overcast", "Temperature": 83.0, "Humidity": 86.0, "Wind": false, "Play": "yes"},
		TrainingItem{"Outlook": "rain", "Temperature": 70.0, "Humidity": 96.0, "Wind": false, "Play": "yes"},
		TrainingItem{"Outlook": "rain", "Temperature": 68.0, "Humidity": 80.0, "Wind": false, "Play": "yes"},
		TrainingItem{"Outlook": "rain", "Temperature": 65.0, "Humidity": 70.0, "Wind": true, "Play": "no"},
		TrainingItem{"Outlook": "overcast", "Temperature": 64.0, "Humidity": 65.0, "Wind": true, "Play": "yes"},
	}
	cfg := Config{CategoryAttr: "Play"}
	model, _ := Train(ts, cfg)

	stats := model.Stats()

	// Verify basic properties
	if stats.TotalNodes == 0 {
		t.Error("expected non-zero total nodes")
	}

	if stats.LeafNodes == 0 {
		t.Error("expected non-zero leaf nodes")
	}

	if stats.InternalNodes == 0 {
		t.Error("expected non-zero internal nodes")
	}

	// Total nodes should equal internal + leaf
	if stats.TotalNodes != stats.LeafNodes+stats.InternalNodes {
		t.Errorf("total nodes (%d) != leaf (%d) + internal (%d)",
			stats.TotalNodes, stats.LeafNodes, stats.InternalNodes)
	}

	// Should have exactly 2 classes
	if len(stats.Classes) != 2 {
		t.Errorf("expected 2 classes, got %d: %v", len(stats.Classes), stats.Classes)
	}

	// Depth should be reasonable (at least 1)
	if stats.TreeDepth < 1 {
		t.Errorf("expected depth >= 1, got %d", stats.TreeDepth)
	}
}

func TestStats_NilModel(t *testing.T) {
	var m *Model
	stats := m.Stats()

	// Should return zero stats
	if stats.TotalNodes != 0 || stats.LeafNodes != 0 || stats.InternalNodes != 0 {
		t.Error("expected zero stats for nil model")
	}
}

func TestStats_NilRoot(t *testing.T) {
	m := &Model{Root: nil, Config: Config{CategoryAttr: "label"}}
	stats := m.Stats()

	// Should return zero stats
	if stats.TotalNodes != 0 || stats.LeafNodes != 0 || stats.InternalNodes != 0 {
		t.Error("expected zero stats for model with nil root")
	}
}

func TestStats_SingleLeaf(t *testing.T) {
	// Model with just a single leaf (all training data has same label)
	ts := TrainingSet{
		TrainingItem{"x": 1.0, "label": "A"},
		TrainingItem{"x": 2.0, "label": "A"},
		TrainingItem{"x": 3.0, "label": "A"},
	}
	cfg := Config{CategoryAttr: "label"}
	model, err := Train(ts, cfg)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	stats := model.Stats()

	if stats.TotalNodes != 1 {
		t.Errorf("expected 1 total node, got %d", stats.TotalNodes)
	}

	if stats.LeafNodes != 1 {
		t.Errorf("expected 1 leaf node, got %d", stats.LeafNodes)
	}

	if stats.InternalNodes != 0 {
		t.Errorf("expected 0 internal nodes, got %d", stats.InternalNodes)
	}

	if stats.TreeDepth != 0 {
		t.Errorf("expected depth 0 (root is leaf), got %d", stats.TreeDepth)
	}

	if len(stats.Classes) != 1 {
		t.Errorf("expected 1 class, got %d", len(stats.Classes))
	}
}

func TestStats_WithMaxDepth(t *testing.T) {
	// Train with depth limit
	ts := TrainingSet{
		TrainingItem{"x": 1.0, "y": 1.0, "label": "A"},
		TrainingItem{"x": 1.0, "y": 2.0, "label": "B"},
		TrainingItem{"x": 2.0, "y": 1.0, "label": "C"},
		TrainingItem{"x": 2.0, "y": 2.0, "label": "D"},
	}
	cfg := Config{CategoryAttr: "label", MaxDepth: 1}
	model, err := Train(ts, cfg)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	stats := model.Stats()

	// With maxDepth=1, tree depth should not exceed 1
	if stats.TreeDepth > 1 {
		t.Errorf("expected depth <= 1 with MaxDepth=1, got %d", stats.TreeDepth)
	}
}
