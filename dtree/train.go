package dtree

import (
    "math"
    "reflect"
)

// Internal helpers

func predicateEq(a, b interface{}) bool { return a == b }

func predicateGte(a, b interface{}) bool {
    switch av := a.(type) {
    case float64:
        return av >= b.(float64)
    case int:
        return float64(av) >= b.(float64)
    case nil:
        // treat missing as unknown; handled at predict time
        return false
    }
    return false
}

func stringInSlice(a string, list []string) bool {
    for _, b := range list {
        if b == a {
            return true
        }
    }
    return false
}

// entropy calculates Shannon entropy (natural log base is fine for comparisons).
func entropy(set TrainingSet, attr string) float64 {
    counter := counterUniqueValues(set, attr)
    var e float64
    total := float64(len(set))
    if total == 0 {
        return 0
    }
    for _, cnt := range counter {
        p := float64(cnt) / total
        e += -p * math.Log(p)
    }
    return e
}

func counterUniqueValues(set TrainingSet, attr string) map[string]int {
    res := make(map[string]int)
    for _, item := range set {
        v := item[attr]
        switch vv := v.(type) {
        case string:
            res[vv] += 1
        case float64:
            res[formatFloatKey(vv)] += 1
        case int:
            res[formatFloatKey(float64(vv))] += 1
        default:
            res["<nil>"] += 1
        }
    }
    return res
}

// Split groups items according to predicate on attr.
type splitResult struct {
    Match         TrainingSet
    NoMatch       TrainingSet
    Gain          float64
    Attribute     string
    Predicate     *Predicate
    PredicateName string
    Pivot         interface{}
}

func split(set TrainingSet, attr string, predicate Predicate, pivot interface{}) splitResult {
    var res splitResult
    for _, item := range set {
        if predicate(item[attr], pivot) {
            res.Match = append(res.Match, item)
        } else {
            res.NoMatch = append(res.NoMatch, item)
        }
    }
    return res
}

// Train builds a decision tree model.
func Train(set TrainingSet, cfg Config) *Model {
    if cfg.Criterion == "" {
        cfg.Criterion = "entropy"
    }
    root := makeTrainingTree(set, cfg, 0)
    return &Model{Root: root, Config: cfg}
}

func makeTrainingTree(set TrainingSet, cfg Config, depth int) *TreeItem {
    // stopping conditions
    if len(set) == 0 {
        return &TreeItem{Category: ""}
    }
    // If pure or thresholds reached -> leaf
    if entropy(set, cfg.CategoryAttr) <= 0.00001 ||
        (cfg.MaxDepth > 0 && depth >= cfg.MaxDepth) ||
        (cfg.MinSamples > 0 && len(set) < cfg.MinSamples) {
        return leafFromSet(set, cfg.CategoryAttr)
    }

    initEntropy := entropy(set, cfg.CategoryAttr)
    var best splitResult

    for _, item := range set {
        for attr, pivot := range item {
            if attr == cfg.CategoryAttr || stringInSlice(attr, cfg.IgnoredAttributes) {
                continue
            }

            var pred Predicate
            var predName string
            // auto-detect numeric vs categorical by pivot type
            if isNumeric(pivot) {
                pred = predicateGte
                predName = ">="
                pivot = toFloat(pivot)
            } else {
                pred = predicateEq
                predName = "=="
            }

            curr := split(set, attr, pred, pivot)
            // information gain
            matchE := entropy(curr.Match, cfg.CategoryAttr)
            noMatchE := entropy(curr.NoMatch, cfg.CategoryAttr)
            newE := (matchE*float64(len(curr.Match)) + noMatchE*float64(len(curr.NoMatch))) / float64(len(set))
            curr.Gain = initEntropy - newE
            curr.Attribute = attr
            curr.Pivot = pivot
            curr.Predicate = &pred
            curr.PredicateName = predName
            if curr.Gain > best.Gain {
                best = curr
            }
        }
    }

    if best.Gain <= 0 {
        return leafFromSet(set, cfg.CategoryAttr)
    }

    return &TreeItem{
        Match:          makeTrainingTree(best.Match, cfg, depth+1),
        NoMatch:        makeTrainingTree(best.NoMatch, cfg, depth+1),
        MatchedCount:   len(best.Match),
        NoMatchedCount: len(best.NoMatch),
        Attribute:      best.Attribute,
        PredicateName:  best.PredicateName,
        Pivot:          best.Pivot,
        ClassCounts:    counterUniqueValues(set, cfg.CategoryAttr),
    }
}

func leafFromSet(set TrainingSet, labelAttr string) *TreeItem {
    counts := counterUniqueValues(set, labelAttr)
    mostVal := mostFrequentValue(counts)
    return &TreeItem{Category: mostVal, ClassCounts: counts}
}

func mostFrequentValue(counts map[string]int) string {
    var bestK string
    var bestV int
    for k, v := range counts {
        if v > bestV {
            bestK, bestV = k, v
        }
    }
    return bestK
}

func isNumeric(v interface{}) bool {
    if v == nil { return false }
    t := reflect.TypeOf(v).Kind()
    return t == reflect.Int || t == reflect.Int32 || t == reflect.Int64 || t == reflect.Float32 || t == reflect.Float64
}

func toFloat(v interface{}) float64 {
    switch vv := v.(type) {
    case int:
        return float64(vv)
    case int32:
        return float64(vv)
    case int64:
        return float64(vv)
    case float32:
        return float64(vv)
    case float64:
        return vv
    }
    return 0
}

func formatFloatKey(f float64) string {
    // coarse formatting to stabilize keys in counts; avoid scientific notation for integers
    if f == math.Trunc(f) {
        return fmtInt(int64(f))
    }
    return trimFloat(f)
}

// minimal helpers to avoid importing fmt
func fmtInt(i int64) string { return itoa(i) }

func itoa(i int64) string {
    // simple base10 itoa
    if i == 0 { return "0" }
    neg := i < 0
    if neg { i = -i }
    buf := make([]byte, 0, 20)
    for i > 0 {
        d := i % 10
        buf = append([]byte{'0' + byte(d)}, buf...)
        i /= 10
    }
    if neg { buf = append([]byte{'-'}, buf...) }
    return string(buf)
}

func trimFloat(f float64) string {
    // round to 6 decimals for keys
    const p = 1e6
    v := math.Round(f*p) / p
    // simple formatter: integer-like if whole
    if v == math.Trunc(v) {
        return fmtInt(int64(v))
    }
    // manually craft with limited decimals
    s := []byte{}
    if v < 0 { s = append(s, '-') ; v = -v }
    ip := int64(math.Trunc(v))
    fp := v - float64(ip)
    s = append(s, []byte(fmtInt(ip))...)
    s = append(s, '.')
    for k := 0; k < 6; k++ {
        fp *= 10
        d := int(fp)
        s = append(s, byte('0'+d))
        fp -= float64(d)
    }
    // trim trailing zeros
    for len(s) > 0 && s[len(s)-1] == '0' { s = s[:len(s)-1] }
    if len(s) > 0 && s[len(s)-1] == '.' { s = s[:len(s)-1] }
    return string(s)
}

