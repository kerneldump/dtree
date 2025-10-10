package dtree

import (
	"encoding/json"
	"errors"
	"io"
	"os"
)

// SaveJSON writes the model to a JSON file.
func (m *Model) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(m)
}

// LoadJSON reads a model from a JSON file and validates it.
func LoadJSON(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return DecodeJSON(f)
}

// DecodeJSON decodes a model from any reader and validates it.
func DecodeJSON(r io.Reader) (*Model, error) {
	dec := json.NewDecoder(r)
	var m Model
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}

	// Validate the loaded model
	if err := m.Validate(); err != nil {
		return nil, err
	}

	return &m, nil
}

// Validate checks if the model is structurally sound and ready for use.
// Returns an error if the model has invalid configuration or tree structure.
func (m *Model) Validate() error {
	if m == nil {
		return errors.New("model is nil")
	}

	if m.Root == nil {
		return errors.New("model has nil root node")
	}

	// Validate configuration
	if m.Config.CategoryAttr == "" {
		return errors.New("model config missing categoryAttr")
	}

	if m.Config.MaxDepth < 0 {
		return errors.New("model config has negative maxDepth")
	}

	if m.Config.MinSamples < 0 {
		return errors.New("model config has negative minSamples")
	}

	// Validate tree structure
	if err := validateNode(m.Root); err != nil {
		return err
	}

	return nil
}

// validateNode recursively checks if a tree node is valid.
func validateNode(node *TreeItem) error {
	if node == nil {
		return nil // nil nodes are allowed as children
	}

	// Check if it's a leaf node
	isLeaf := node.Match == nil && node.NoMatch == nil

	if isLeaf {
		// Leaf nodes must have class counts
		if node.ClassCounts == nil {
			return errors.New("leaf node missing classCounts")
		}
		// Category can be empty string, so we don't validate it
		return nil
	}

	// Internal nodes must have both children
	if node.Match == nil || node.NoMatch == nil {
		return errors.New("internal node missing one or both children")
	}

	// Internal nodes must have split metadata
	if node.Attribute == "" {
		return errors.New("internal node missing attribute")
	}

	if node.PredicateName == "" {
		return errors.New("internal node missing predicateName")
	}

	// Validate predicate name
	if node.PredicateName != "==" && node.PredicateName != ">=" {
		return errors.New("internal node has invalid predicateName (must be == or >=)")
	}

	// Internal nodes should have class counts for fallback prediction
	if node.ClassCounts == nil {
		return errors.New("internal node missing classCounts")
	}

	// Recursively validate children
	if err := validateNode(node.Match); err != nil {
		return err
	}

	if err := validateNode(node.NoMatch); err != nil {
		return err
	}

	return nil
}
