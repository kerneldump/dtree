package dtree



// calculateProba is a helper to compute probabilities from a class counts map.
func calculateProba(counts map[string]int) map[string]float64 {
	total := 0
	for _, c := range counts {
		total += c
	}
	out := make(map[string]float64, len(counts))
	if total == 0 {
		return out
	}
	for k, v := range counts {
		out[k] = float64(v) / float64(total)
	}
	return out
}

// Predict returns the hard class prediction for an item.
func (m *Model) Predict(item TrainingItem) string {
    node := m.Root
    for node != nil {
        // Leaf detection should be structural only; labels may be empty strings.
        if node.Match == nil && node.NoMatch == nil {
            return node.Category
        }

        // Decide which child to visit next.
        var nextNode *TreeItem
        val, ok := item[node.Attribute]

        if !ok { // attribute truly missing
            if node.MatchedCount >= node.NoMatchedCount {
                nextNode = node.Match
            } else {
                nextNode = node.NoMatch
            }
        } else {
            // Attribute present; handle comparator specifics.
            var goMatch bool
            if node.PredicateName == ">=" {
                // For numeric comparator, treat nil value as missing.
                if val == nil {
                    if node.MatchedCount >= node.NoMatchedCount {
                        nextNode = node.Match
                    } else {
                        nextNode = node.NoMatch
                    }
                } else {
                    goMatch = predicateGte(toComparable(val), node.Pivot)
                    if goMatch { nextNode = node.Match } else { nextNode = node.NoMatch }
                }
            } else { // equality comparator "=="
                // Evaluate equality even if val == nil so that nil==nil can match.
                goMatch = predicateEq(val, node.Pivot)
                if goMatch { nextNode = node.Match } else { nextNode = node.NoMatch }
            }
        }

        // If the next step is a dead end, predict using the current node's majority class.
        if nextNode == nil {
            return mostFrequentValue(node.ClassCounts)
        }
        node = nextNode
    }
    return "" // Only if root is nil
}

// PredictProba returns class probabilities at the reached leaf.
func (m *Model) PredictProba(item TrainingItem) map[string]float64 {
    node := m.Root
    for node != nil {
        // Leaf detection should be structural only.
        if node.Match == nil && node.NoMatch == nil {
            return calculateProba(node.ClassCounts)
        }

        // Decide which child to visit next.
        var nextNode *TreeItem
        val, ok := item[node.Attribute]

        if !ok { // attribute truly missing
            if node.MatchedCount >= node.NoMatchedCount {
                nextNode = node.Match
            } else {
                nextNode = node.NoMatch
            }
        } else {
            // Attribute present; handle comparator specifics.
            var goMatch bool
            if node.PredicateName == ">=" {
                if val == nil {
                    if node.MatchedCount >= node.NoMatchedCount {
                        nextNode = node.Match
                    } else {
                        nextNode = node.NoMatch
                    }
                } else {
                    goMatch = predicateGte(toComparable(val), node.Pivot)
                    if goMatch { nextNode = node.Match } else { nextNode = node.NoMatch }
                }
            } else { // equality comparator
                goMatch = predicateEq(val, node.Pivot)
                if goMatch { nextNode = node.Match } else { nextNode = node.NoMatch }
            }
        }

        // If the next step is a dead end, predict using the current node's probabilities.
        if nextNode == nil {
            return calculateProba(node.ClassCounts)
        }
        node = nextNode
    }
    return map[string]float64{} // Only if root is nil
}

// Batch helpers
func (m *Model) PredictBatch(items []TrainingItem) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = m.Predict(it)
	}
	return out
}

func (m *Model) PredictProbaBatch(items []TrainingItem) []map[string]float64 {
	out := make([]map[string]float64, len(items))
	for i, it := range items {
		out[i] = m.PredictProba(it)
	}
	return out
}

// normalize numeric values to float64 for comparison
func toComparable(v interface{}) interface{} {
	if isNumeric(v) {
		return toFloat(v)
	}
	return v
}
