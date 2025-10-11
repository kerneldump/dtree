package dtree

// TrainingItem represents a single row with arbitrary attributes.
// Values may be string or numeric (int/float64). Numeric detection is automatic.
type TrainingItem map[string]interface{}

// TrainingSet is a collection of training items.
type TrainingSet []TrainingItem

// Config controls tree training and behavior.
type Config struct {
	// CategoryAttr is the label/target attribute to predict (required).
	CategoryAttr string `json:"categoryAttr"`
	// IgnoredAttributes will be excluded when searching for splits.
	IgnoredAttributes []string `json:"ignoredAttributes,omitempty"`
	// Criterion selects the split criterion. Currently supports "entropy" only.
	Criterion string `json:"criterion,omitempty"`
	// MaxDepth limits the depth of the tree. 0 means unlimited.
	MaxDepth int `json:"maxDepth,omitempty"`
	// MinSamples stops splitting when a node has fewer than MinSamples. 0 means no limit.
	MinSamples int `json:"minSamples,omitempty"`
}

// Model wraps a trained tree and training configuration.
type Model struct {
	Root   *TreeItem `json:"root"`
	Config Config    `json:"config"`
}

// ModelStats contains statistics about a trained model.
type ModelStats struct {
	// TreeDepth is the maximum depth of the tree (distance from root to deepest leaf)
	TreeDepth int
	// TotalNodes is the total number of nodes (internal + leaf)
	TotalNodes int
	// LeafNodes is the number of leaf nodes
	LeafNodes int
	// InternalNodes is the number of internal (decision) nodes
	InternalNodes int
	// Classes is the set of unique class labels found in leaf nodes
	Classes []string
}

// Predicate compares an item's value against the pivot, returning true to go to Match branch.
type Predicate func(interface{}, interface{}) bool

// TreeItem is a node in the decision tree.
type TreeItem struct {
	// Tree structure
	Match   *TreeItem `json:"match,omitempty"`
	NoMatch *TreeItem `json:"noMatch,omitempty"`

	// Predicted category at leaf (most frequent label)
	Category string `json:"category,omitempty"`
	// ClassCounts at leaf for probability output
	ClassCounts map[string]int `json:"classCounts,omitempty"`

	// Split metadata
	MatchedCount   int         `json:"matchedCount,omitempty"`
	NoMatchedCount int         `json:"noMatchedCount,omitempty"`
	Attribute      string      `json:"attribute,omitempty"`
	PredicateName  string      `json:"predicateName,omitempty"`
	Pivot          interface{} `json:"pivot,omitempty"`
}
