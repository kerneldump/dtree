package dtree

import "errors"

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
// Returns an error if the model is invalid or prediction fails.
func (m *Model) Predict(item TrainingItem) (string, error) {
	if m == nil {
		return "", errors.New("model is nil")
	}
	if m.Root == nil {
		return "", errors.New("model has nil root node")
	}
	if item == nil {
		return "", errors.New("item cannot be nil")
	}

	node := m.Root
	for node != nil {
		// Leaf detection should be structural only; labels may be empty strings.
		if node.Match == nil && node.NoMatch == nil {
			return node.Category, nil
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
					if goMatch {
						nextNode = node.Match
					} else {
						nextNode = node.NoMatch
					}
				}
			} else { // equality comparator "=="
				// Evaluate equality even if val == nil so that nil==nil can match.
				goMatch = predicateEq(val, node.Pivot)
				if goMatch {
					nextNode = node.Match
				} else {
					nextNode = node.NoMatch
				}
			}
		}

		// If the next step is a dead end, predict using the current node's majority class.
		if nextNode == nil {
			return mostFrequentValue(node.ClassCounts), nil
		}
		node = nextNode
	}

	// Should never reach here if model is valid
	return "", errors.New("reached end of tree without finding leaf node")
}

// PredictProba returns class probabilities at the reached leaf.
// Returns an error if the model is invalid or prediction fails.
func (m *Model) PredictProba(item TrainingItem) (map[string]float64, error) {
	if m == nil {
		return nil, errors.New("model is nil")
	}
	if m.Root == nil {
		return nil, errors.New("model has nil root node")
	}
	if item == nil {
		return nil, errors.New("item cannot be nil")
	}

	node := m.Root
	for node != nil {
		// Leaf detection should be structural only.
		if node.Match == nil && node.NoMatch == nil {
			return calculateProba(node.ClassCounts), nil
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
					if goMatch {
						nextNode = node.Match
					} else {
						nextNode = node.NoMatch
					}
				}
			} else { // equality comparator
				goMatch = predicateEq(val, node.Pivot)
				if goMatch {
					nextNode = node.Match
				} else {
					nextNode = node.NoMatch
				}
			}
		}

		// If the next step is a dead end, predict using the current node's probabilities.
		if nextNode == nil {
			return calculateProba(node.ClassCounts), nil
		}
		node = nextNode
	}

	// Should never reach here if model is valid
	return nil, errors.New("reached end of tree without finding leaf node")
}

// PredictBatch predicts classes for multiple items.
// Returns predictions and an error if any prediction fails.
// On error, returns partial results up to the point of failure.
func (m *Model) PredictBatch(items []TrainingItem) ([]string, error) {
	out := make([]string, len(items))
	for i, it := range items {
		pred, err := m.Predict(it)
		if err != nil {
			return out[:i], err
		}
		out[i] = pred
	}
	return out, nil
}

// PredictProbaBatch predicts class probabilities for multiple items.
// Returns probabilities and an error if any prediction fails.
// On error, returns partial results up to the point of failure.
func (m *Model) PredictProbaBatch(items []TrainingItem) ([]map[string]float64, error) {
	out := make([]map[string]float64, len(items))
	for i, it := range items {
		proba, err := m.PredictProba(it)
		if err != nil {
			return out[:i], err
		}
		out[i] = proba
	}
	return out, nil
}

// normalize numeric values to float64 for comparison
func toComparable(v interface{}) interface{} {
	if isNumeric(v) {
		return toFloat(v)
	}
	return v
}
