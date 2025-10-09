package dtree

import "testing"

func TestCounterUnique(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"param": "yes"},
		TrainingItem{"param": "no"},
		TrainingItem{"param": "yes"},
		TrainingItem{"param": "yes"},
	}
	vals := counterUniqueValues(ts, "param")
	if vals["yes"] != 3 || vals["no"] != 1 {
		t.Fatalf("unexpected counts: %+v", vals)
	}
}

func TestEntropyPositive(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"param": "yes"},
		TrainingItem{"param": "no"},
		TrainingItem{"param": "yes"},
		TrainingItem{"param": "yes"},
	}
	if v := entropy(ts, "param"); !(v > 0) {
		t.Fatalf("entropy should be > 0, got %v", v)
	}
}

func TestSplit(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"param": "yes", "age": 20.0},
		TrainingItem{"param": "no", "age": 30.0},
		TrainingItem{"param": "yes", "age": 1.0},
	}
	res := split(ts, "age", predicateGte, 20.0)
	if len(res.Match) != 2 || len(res.NoMatch) != 1 {
		t.Fatalf("unexpected split sizes: %+v", res)
	}
}

func TestTrainAndPredict_PlayTennis(t *testing.T) {
	// Small subset of the PlayTennis dataset
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
	model, err := Train(ts, cfg)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}
	// Predict on an unseen but similar example where overcast tends to be yes
	item := TrainingItem{"Outlook": "overcast", "Temperature": 72.0, "Humidity": 90.0, "Wind": true}
	if got := model.Predict(item); got != "yes" {
		t.Fatalf("expected yes, got %s", got)
	}
}

// New validation tests

func TestTrain_EmptyTrainingSet(t *testing.T) {
	ts := TrainingSet{}
	cfg := Config{CategoryAttr: "label"}
	_, err := Train(ts, cfg)
	if err == nil {
		t.Fatal("expected error for empty training set")
	}
	if err.Error() != "training set cannot be empty" {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestTrain_MissingCategoryAttr(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"feature": "value"},
	}
	cfg := Config{CategoryAttr: ""}
	_, err := Train(ts, cfg)
	if err == nil {
		t.Fatal("expected error for missing categoryAttr")
	}
	if err.Error() != "config.CategoryAttr is required" {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestTrain_CategoryAttrNotFound(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"feature": "value"},
	}
	cfg := Config{CategoryAttr: "nonexistent"}
	_, err := Train(ts, cfg)
	if err == nil {
		t.Fatal("expected error for nonexistent categoryAttr")
	}
	if err.Error() != "categoryAttr not found in any training items" {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestTrain_NegativeMaxDepth(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"label": "yes"},
	}
	cfg := Config{CategoryAttr: "label", MaxDepth: -1}
	_, err := Train(ts, cfg)
	if err == nil {
		t.Fatal("expected error for negative maxDepth")
	}
	if err.Error() != "config.MaxDepth cannot be negative" {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestTrain_NegativeMinSamples(t *testing.T) {
	ts := TrainingSet{
		TrainingItem{"label": "yes"},
	}
	cfg := Config{CategoryAttr: "label", MinSamples: -1}
	_, err := Train(ts, cfg)
	if err == nil {
		t.Fatal("expected error for negative minSamples")
	}
	if err.Error() != "config.MinSamples cannot be negative" {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestPredicateGte_SafeTypeAssertion(t *testing.T) {
	// Test that predicateGte doesn't panic with invalid types
	result := predicateGte("not a number", 10.0)
	if result != false {
		t.Fatalf("expected false for invalid type comparison")
	}
}
