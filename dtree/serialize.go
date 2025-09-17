package dtree

import (
    "encoding/json"
    "io"
    "os"
)

// SaveJSON writes the model to a JSON file.
func (m *Model) SaveJSON(path string) error {
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    enc := json.NewEncoder(f)
    enc.SetIndent("", "  ")
    return enc.Encode(m)
}

// LoadJSON reads a model from a JSON file.
func LoadJSON(path string) (*Model, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()
    return DecodeJSON(f)
}

// DecodeJSON decodes a model from any reader.
func DecodeJSON(r io.Reader) (*Model, error) {
    dec := json.NewDecoder(r)
    var m Model
    if err := dec.Decode(&m); err != nil { return nil, err }
    return &m, nil
}

