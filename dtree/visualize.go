package dtree

import (
    "fmt"
    "html/template"
    "os"
)

const htmlTemplate = `<html>
<head>
<style type="text/css">
  * { margin: 0; padding: 0; }
  .tree ul { padding-top: 20px; position: relative; }
  .tree li { white-space: nowrap; float: left; text-align: center; list-style-type: none; position: relative; padding: 20px 5px 0 5px; }
  .tree li::before, .tree li::after{ content: ''; position: absolute; top: 0; right: 50%; border-top: 1px solid #ccc; width: 50%; height: 20px; }
  .tree li::after{ right: auto; left: 50%; border-left: 1px solid #ccc; }
  .tree li:only-child::after, .tree li:only-child::before { display: none; }
  .tree li:only-child{ padding-top: 0; }
  .tree li:first-child::before, .tree li:last-child::after{ border: 0 none; }
  .tree li:last-child::before{ border-right: 1px solid #ccc; border-radius: 0 5px 0 0; }
  .tree li:first-child::after{ border-radius: 5px 0 0 0; }
  .tree ul ul::before{ content: ''; position: absolute; top: 0; left: 50%; border-left: 1px solid #ccc; width: 0; height: 20px; }
  .tree li a{ border: 1px solid #ccc; padding: 5px 10px; text-decoration: none; color: #666; font-family: arial, verdana, tahoma; font-size: 11px; display: inline-block; border-radius: 5px; }
</style>
</head>
<body>
<div class="tree">{{ .tree }}</div>
</body>
</html>`

// ToHTML writes a simple interactive HTML rendering of the tree.
func (m *Model) ToHTML(path string) error {
    tmpl, err := template.New("tree").Parse(htmlTemplate)
    if err != nil { return err }
    f, err := os.Create(path)
    if err != nil { return err }
    defer f.Close()
    data := map[string]template.HTML{"tree": template.HTML(treeToHTML(m.Root))}
    return tmpl.Execute(f, data)
}

func treeToHTML(node *TreeItem) string {
    if node == nil { return "" }
    if node.Category != "" && node.Match == nil && node.NoMatch == nil {
        return `<ul><li><a href="#"><b>` + node.Category + `</b></a></li></ul>`
    }
    return `<ul>
      <li><a href="#"><b>` + fmt.Sprintf("%s %s %v", node.Attribute, node.PredicateName, node.Pivot) + `</b></a>
        <ul>
          <li><a href="#">yes</a>` + treeToHTML(node.Match) + `</li>
          <li><a href="#">no</a>` + treeToHTML(node.NoMatch) + `</li>
        </ul>
      </li>
    </ul>`
}

// ToDOT writes a Graphviz DOT representation.
func (m *Model) ToDOT() string {
    b := &dotBuilder{next: 0}
    b.line("digraph dtree {")
    b.line("  node [shape=box];")
    b.walk(m.Root)
    b.line("}")
    return b.buf
}

type dotBuilder struct {
    next int
    buf  string
}

func (d *dotBuilder) id() int { d.next++; return d.next }
func (d *dotBuilder) line(s string) { d.buf += s + "\n" }

func (d *dotBuilder) walk(n *TreeItem) int {
    if n == nil { return -1 }
    id := d.id()
    if n.Category != "" && n.Match == nil && n.NoMatch == nil {
        d.line(fmt.Sprintf("  n%d [label=\"%s\", shape=oval];", id, n.Category))
        return id
    }
    d.line(fmt.Sprintf("  n%d [label=\"%s %s %v\"];", id, n.Attribute, n.PredicateName, n.Pivot))
    lm := d.walk(n.Match)
    ln := d.walk(n.NoMatch)
    if lm != -1 { d.line(fmt.Sprintf("  n%d -> n%d [label=\"yes\"];", id, lm)) }
    if ln != -1 { d.line(fmt.Sprintf("  n%d -> n%d [label=\"no\"];", id, ln)) }
    return id
}

