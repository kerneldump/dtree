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
        TrainingItem{"Outlook": "sunny",    "Temperature": 85.0, "Humidity": 85.0, "Wind": false, "Play": "no"},
        TrainingItem{"Outlook": "sunny",    "Temperature": 80.0, "Humidity": 90.0, "Wind": true,  "Play": "no"},
        TrainingItem{"Outlook": "overcast", "Temperature": 83.0, "Humidity": 86.0, "Wind": false, "Play": "yes"},
        TrainingItem{"Outlook": "rain",     "Temperature": 70.0, "Humidity": 96.0, "Wind": false, "Play": "yes"},
        TrainingItem{"Outlook": "rain",     "Temperature": 68.0, "Humidity": 80.0, "Wind": false, "Play": "yes"},
        TrainingItem{"Outlook": "rain",     "Temperature": 65.0, "Humidity": 70.0, "Wind": true,  "Play": "no"},
        TrainingItem{"Outlook": "overcast", "Temperature": 64.0, "Humidity": 65.0, "Wind": true,  "Play": "yes"},
    }
    cfg := Config{CategoryAttr: "Play"}
    model := Train(ts, cfg)
    // Predict on an unseen but similar example where overcast tends to be yes
    item := TrainingItem{"Outlook": "overcast", "Temperature": 72.0, "Humidity": 90.0, "Wind": true}
    if got := model.Predict(item); got != "yes" {
        t.Fatalf("expected yes, got %s", got)
    }
}

