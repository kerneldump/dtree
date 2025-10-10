package dtree

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestValidate_ValidModel(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"feature": "a", "label": "yes"},
		TrainingItem{"feature": "b", "label": "no"},
	}
	cfg := Config{CategoryAttr: "label"}
	model, _ := Train(ts, cfg)

	if err := model.Validate(); err != nil {
		t.Fatalf("valid model failed validation: %v", err)
	}
}

func TestValidate_NilModel(t *testing.T) {
	var m *Model
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for nil model")
	}
	if err.Error() != "model is nil" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_NilRoot(t *testing.T) {
	m := &Model{
		Root:   nil,
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for nil root")
	}
	if err.Error() != "model has nil root node" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_MissingCategoryAttr(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Category:    "yes",
			ClassCounts: map[string]int{"yes": 1},
		},
		Config: Config{CategoryAttr: ""},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for missing categoryAttr")
	}
	if err.Error() != "model config missing categoryAttr" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_NegativeMaxDepth(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Category:    "yes",
			ClassCounts: map[string]int{"yes": 1},
		},
		Config: Config{CategoryAttr: "label", MaxDepth: -1},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for negative maxDepth")
	}
	if err.Error() != "model config has negative maxDepth" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_NegativeMinSamples(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Category:    "yes",
			ClassCounts: map[string]int{"yes": 1},
		},
		Config: Config{CategoryAttr: "label", MinSamples: -1},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for negative minSamples")
	}
	if err.Error() != "model config has negative minSamples" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_LeafMissingClassCounts(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Category:    "yes",
			ClassCounts: nil, // Invalid: leaf must have class counts
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for leaf missing classCounts")
	}
	if err.Error() != "leaf node missing classCounts" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_InternalNodeMissingChild(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Attribute:     "feature",
			PredicateName: "==",
			Pivot:         "a",
			ClassCounts:   map[string]int{"yes": 1, "no": 1},
			Match: &TreeItem{
				Category:    "yes",
				ClassCounts: map[string]int{"yes": 1},
			},
			NoMatch: nil, // Invalid: internal node missing NoMatch child
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for internal node missing child")
	}
	if err.Error() != "internal node missing one or both children" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_InternalNodeMissingAttribute(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Attribute:     "", // Invalid: internal node missing attribute
			PredicateName: "==",
			Pivot:         "a",
			ClassCounts:   map[string]int{"yes": 1, "no": 1},
			Match: &TreeItem{
				Category:    "yes",
				ClassCounts: map[string]int{"yes": 1},
			},
			NoMatch: &TreeItem{
				Category:    "no",
				ClassCounts: map[string]int{"no": 1},
			},
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for internal node missing attribute")
	}
	if err.Error() != "internal node missing attribute" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_InternalNodeMissingPredicateName(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Attribute:     "feature",
			PredicateName: "", // Invalid: internal node missing predicateName
			Pivot:         "a",
			ClassCounts:   map[string]int{"yes": 1, "no": 1},
			Match: &TreeItem{
				Category:    "yes",
				ClassCounts: map[string]int{"yes": 1},
			},
			NoMatch: &TreeItem{
				Category:    "no",
				ClassCounts: map[string]int{"no": 1},
			},
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for internal node missing predicateName")
	}
	if err.Error() != "internal node missing predicateName" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_InternalNodeInvalidPredicateName(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Attribute:     "feature",
			PredicateName: "!=", // Invalid: only == and >= are allowed
			Pivot:         "a",
			ClassCounts:   map[string]int{"yes": 1, "no": 1},
			Match: &TreeItem{
				Category:    "yes",
				ClassCounts: map[string]int{"yes": 1},
			},
			NoMatch: &TreeItem{
				Category:    "no",
				ClassCounts: map[string]int{"no": 1},
			},
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for invalid predicateName")
	}
	if !strings.Contains(err.Error(), "invalid predicateName") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_InternalNodeMissingClassCounts(t *testing.T) {
	m := &Model{
		Root: &TreeItem{
			Attribute:     "feature",
			PredicateName: "==",
			Pivot:         "a",
			ClassCounts:   nil, // Invalid: internal node needs class counts for fallback
			Match: &TreeItem{
				Category:    "yes",
				ClassCounts: map[string]int{"yes": 1},
			},
			NoMatch: &TreeItem{
				Category:    "no",
				ClassCounts: map[string]int{"no": 1},
			},
		},
		Config: Config{CategoryAttr: "label"},
	}
	err := m.Validate()
	if err == nil {
		t.Fatal("expected error for internal node missing classCounts")
	}
	if err.Error() != "internal node missing classCounts" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDecodeJSON_InvalidModel(t *testing.T) {
	// Create a JSON with invalid structure
	invalidJSON := `{
		"root": {
			"match": {
				"category": "yes",
				"classCounts": null
			},
			"noMatch": null,
			"attribute": "feature",
			"predicateName": "==",
			"pivot": "a",
			"classCounts": {"yes": 1}
		},
		"config": {
			"categoryAttr": "label"
		}
	}`

	r := bytes.NewReader([]byte(invalidJSON))
	_, err := DecodeJSON(r)
	if err == nil {
		t.Fatal("expected error when decoding invalid model")
	}
}

func TestLoadJSON_ValidatesModel(t *testing.T) {
	// Create a valid model
	ts := TrainingSet{
		TrainingItem{"feature": "a", "label": "yes"},
		TrainingItem{"feature": "b", "label": "no"},
	}
	cfg := Config{CategoryAttr: "label"}
	model, _ := Train(ts, cfg)

	// Save it
	tmpFile := "test_model.json"
	if err := model.SaveJSON(tmpFile); err != nil {
		t.Fatalf("failed to save model: %v", err)
	}
	defer func() {
		// Clean up
		_ = os.Remove(tmpFile)
	}()

	// Load it - should validate automatically
	loaded, err := LoadJSON(tmpFile)
	if err != nil {
		t.Fatalf("failed to load valid model: %v", err)
	}

	if loaded == nil {
		t.Fatal("loaded model is nil")
	}
}

func TestLoadJSON_RejectsInvalidModel(t *testing.T) {
	// Create an invalid model JSON
	invalidJSON := `{
		"root": null,
		"config": {
			"categoryAttr": "label"
		}
	}`

	tmpFile := "test_invalid_model.json"
	if err := os.WriteFile(tmpFile, []byte(invalidJSON), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}
	defer func() {
		_ = os.Remove(tmpFile)
	}()

	// Attempt to load - should fail validation
	_, err := LoadJSON(tmpFile)
	if err == nil {
		t.Fatal("expected error when loading invalid model")
	}

	if !strings.Contains(err.Error(), "nil root") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSaveJSON_RoundTrip(t *testing.T) {
	// Create a model
	ts := TrainingSet{
		TrainingItem{"x": 1.0, "y": "a", "label": "A"},
		TrainingItem{"x": 2.0, "y": "b", "label": "B"},
	}
	cfg := Config{CategoryAttr: "label", MaxDepth: 5, MinSamples: 2}
	original, _ := Train(ts, cfg)

	// Save it
	tmpFile := "test_roundtrip.json"
	if err := original.SaveJSON(tmpFile); err != nil {
		t.Fatalf("failed to save: %v", err)
	}
	defer os.Remove(tmpFile)

	// Load it
	loaded, err := LoadJSON(tmpFile)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	// Verify configuration preserved
	if loaded.Config.CategoryAttr != original.Config.CategoryAttr {
		t.Error("categoryAttr not preserved")
	}
	if loaded.Config.MaxDepth != original.Config.MaxDepth {
		t.Error("maxDepth not preserved")
	}
	if loaded.Config.MinSamples != original.Config.MinSamples {
		t.Error("minSamples not preserved")
	}

	// Verify it can predict
	testItem := TrainingItem{"x": 1.5, "y": "a"}
	pred1, err1 := original.Predict(testItem)
	pred2, err2 := loaded.Predict(testItem)

	if err1 != nil || err2 != nil {
		t.Fatal("prediction failed on original or loaded model")
	}

	if pred1 != pred2 {
		t.Errorf("predictions differ: original=%s, loaded=%s", pred1, pred2)
	}
}
