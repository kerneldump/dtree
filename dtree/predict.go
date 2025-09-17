package dtree

// Predict returns the hard class prediction for an item.
func (m *Model) Predict(item TrainingItem) string {
    node := m.Root
    for node != nil {
        if node.Category != "" && node.Match == nil && node.NoMatch == nil {
            return node.Category
        }
        // internal node
        val, ok := item[node.Attribute]
        if !ok || val == nil {
            // missing: route to larger child
            if node.MatchedCount >= node.NoMatchedCount {
                node = node.Match
            } else {
                node = node.NoMatch
            }
            continue
        }
        // evaluate predicate
        var goMatch bool
        if node.PredicateName == ">=" {
            goMatch = predicateGte(toComparable(val), node.Pivot)
        } else {
            goMatch = predicateEq(val, node.Pivot)
        }
        if goMatch {
            node = node.Match
        } else {
            node = node.NoMatch
        }
    }
    return ""
}

// PredictProba returns class probabilities at the reached leaf.
func (m *Model) PredictProba(item TrainingItem) map[string]float64 {
    node := m.Root
    for node != nil {
        if node.ClassCounts != nil && (node.Match == nil && node.NoMatch == nil) {
            total := 0
            for _, c := range node.ClassCounts { total += c }
            out := make(map[string]float64, len(node.ClassCounts))
            if total == 0 { return out }
            for k, v := range node.ClassCounts {
                out[k] = float64(v) / float64(total)
            }
            return out
        }
        val, ok := item[node.Attribute]
        if !ok || val == nil {
            if node.MatchedCount >= node.NoMatchedCount {
                node = node.Match
            } else {
                node = node.NoMatch
            }
            continue
        }
        var goMatch bool
        if node.PredicateName == ">=" {
            goMatch = predicateGte(toComparable(val), node.Pivot)
        } else {
            goMatch = predicateEq(val, node.Pivot)
        }
        if goMatch { node = node.Match } else { node = node.NoMatch }
    }
    return map[string]float64{}
}

// Batch helpers
func (m *Model) PredictBatch(items []TrainingItem) []string {
    out := make([]string, len(items))
    for i, it := range items { out[i] = m.Predict(it) }
    return out
}

func (m *Model) PredictProbaBatch(items []TrainingItem) []map[string]float64 {
    out := make([]map[string]float64, len(items))
    for i, it := range items { out[i] = m.PredictProba(it) }
    return out
}

// normalize numeric values to float64 for comparison
func toComparable(v interface{}) interface{} {
    if isNumeric(v) { return toFloat(v) }
    return v
}

