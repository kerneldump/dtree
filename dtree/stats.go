package dtree

// Stats computes and returns statistics about the model's tree structure.
// This is useful for understanding model complexity and debugging.
func (m *Model) Stats() ModelStats {
	if m == nil || m.Root == nil {
		return ModelStats{}
	}

	stats := ModelStats{}
	classSet := make(map[string]bool)

	// Recursively collect statistics
	collectStats(m.Root, 0, &stats, classSet)

	// Convert class set to sorted slice
	for class := range classSet {
		stats.Classes = append(stats.Classes, class)
	}

	return stats
}

// collectStats recursively traverses the tree and collects statistics.
func collectStats(node *TreeItem, depth int, stats *ModelStats, classSet map[string]bool) {
	if node == nil {
		return
	}

	// Update total nodes
	stats.TotalNodes++

	// Update max depth
	if depth > stats.TreeDepth {
		stats.TreeDepth = depth
	}

	// Check if it's a leaf
	isLeaf := node.Match == nil && node.NoMatch == nil

	if isLeaf {
		stats.LeafNodes++
		// Collect class from leaf
		if node.Category != "" {
			classSet[node.Category] = true
		}
	} else {
		stats.InternalNodes++
		// Recurse to children
		collectStats(node.Match, depth+1, stats, classSet)
		collectStats(node.NoMatch, depth+1, stats, classSet)
	}
}
