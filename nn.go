package main

import (
	"fmt"
	"math"
	"math/rand"
)

var r = rand.New(rand.NewSource(60000))

type Value struct {
	Data     float64
	Label    string
	Operator string
	Gradient float64
	Children []*Value
	Backward func()
}

func NewValue(data float64, label string) *Value {
	return &Value{Data: data, Label: label, Backward: func() {}, Gradient: 0}
}

func (v *Value) String() string {
	s := fmt.Sprintf("Value(data=%.2f)", v.Data)
	if v.Operator != "" {
		s += fmt.Sprintf(" operator=%s", v.Operator)
	}

	if len(v.Children) > 0 {
		for i, c := range v.Children {
			s += fmt.Sprintf(", child%d=%.2f", i, c.Data)
		}
	}

	s += fmt.Sprintf(", gradient=%.2f", v.Gradient)

	return s
}

func (v *Value) Add(arg interface{}) *Value {
	aV, ok := arg.(*Value)
	if !ok {
		aV = NewValue(arg.(float64), "")
	}
	out := &Value{Data: v.Data + aV.Data, Operator: "+", Children: []*Value{v, aV}}
	out.Backward = func() {
		v.Gradient += out.Gradient * 1
		aV.Gradient += out.Gradient * 1
	}
	return out
}

func (v *Value) Subtract(arg interface{}) *Value {
	aV, ok := arg.(*Value)
	if !ok {
		aV = NewValue(arg.(float64), "")
	}
	aV.Data *= -1

	return v.Add(aV)
}

func (v *Value) Multiply(arg interface{}) *Value {
	aV, ok := arg.(*Value)
	if !ok {
		aV = NewValue(arg.(float64), "")
	}
	out := &Value{Data: v.Data * aV.Data, Operator: "*", Children: []*Value{v, aV}}
	out.Backward = func() {
		v.Gradient += aV.Data * out.Gradient
		aV.Gradient += v.Data * out.Gradient
	}
	return out
}

func (v *Value) Exp() *Value {
	out := &Value{Data: math.Exp(v.Data), Operator: "exp", Children: []*Value{v}}
	out.Backward = func() {
		v.Gradient += out.Data * out.Gradient
	}
	return out
}

func (v *Value) Division(arg interface{}) *Value {
	aV, ok := arg.(*Value)
	if !ok {
		aV = NewValue(arg.(float64), "")
	}
	return v.Multiply(aV.Power(-1.0))
}

func (v *Value) Power(other float64) *Value {
	out := &Value{Data: math.Pow(v.Data, other), Operator: "**", Children: []*Value{v}}
	out.Backward = func() {
		v.Gradient += (other * math.Pow(v.Data, other-1)) * out.Gradient // Gradient * Power rule
	}
	return out
}

func (v *Value) Tanh() *Value {
	out := &Value{Data: math.Tanh(v.Data), Operator: "Tanh", Children: []*Value{v}}
	out.Backward = func() {
		v.Gradient += (1 - math.Pow(out.Data, 2)) * out.Gradient
	}
	return out
}

func (v *Value) SetLabel(label string) *Value {
	v.Label = label
	return v
}

func (v *Value) RecursiveBackward() {
	nodes := []*Value{}
	visited := map[*Value]bool{}

	var topologicalSort func(v *Value)
	topologicalSort = func(v *Value) {
		if _, ok := visited[v]; !ok {
			visited[v] = true
			for _, c := range v.Children {
				topologicalSort(c)
			}
			nodes = append(nodes, v)
		}
	}

	topologicalSort(v)

	v.Gradient = 1.0
	for i := len(nodes) - 1; i >= 0; i-- {
		nodes[i].Backward()
	}
}

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(nin int) *Neuron {
	weights := make([]*Value, 0)
	for i := 0; i < nin; i++ {
		weights = append(weights, NewValue(randomFloat(), fmt.Sprintf("Weight:%d", i)))
	}
	return &Neuron{
		Weights: weights,
		Bias:    NewValue(0.0, "Bias"),
	}
}

func (n *Neuron) String() string {
	top := fmt.Sprintf("(%d Weights). Bias(%s)\n", len(n.Weights), n.Bias)
	for i, w := range n.Weights {
		top += fmt.Sprintf("\tWeight(%d) - %s\n", i, w)
	}

	return top
}

func (n *Neuron) Call(x []*Value) *Value {
	sum := NewValue(n.Bias.Data, "Sum")
	for i, weight := range n.Weights {
		sum = sum.Add(weight.Multiply(x[i]))
	}

	return sum.Tanh().SetLabel("tanh")
}

func (n *Neuron) Parameters() []*Value {
	p := make([]*Value, 0)
	p = append(p, n.Bias)
	p = append(p, n.Weights...)

	return p
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin, nout int) *Layer {
	neurons := make([]*Neuron, 0)
	for i := 0; i < nout; i++ {
		neurons = append(neurons, NewNeuron(nin))
	}
	return &Layer{neurons}
}

func (l *Layer) String() string {
	s := fmt.Sprintf("Layer (%d Neurons)\n", len(l.Neurons))

	for i, n := range l.Neurons {
		s += fmt.Sprintf("\tNeuron:%d - %s\n", i, n)
	}

	return s
}

func (l *Layer) Call(x []*Value) []*Value {
	outs := make([]*Value, 0)
	for _, n := range l.Neurons {
		outs = append(outs, n.Call(x))
	}
	return outs
}

func (l *Layer) Parameters() []*Value {
	out := make([]*Value, 0)
	for _, n := range l.Neurons {
		out = append(out, n.Parameters()...)
	}
	return out
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := []int{nin}
	sz = append(sz, nouts...)

	layers := make([]*Layer, 0)
	for i := 0; i < len(nouts); i++ {
		layers = append(layers, NewLayer(sz[i], sz[i+1]))
	}

	return &MLP{layers}
}

func (m *MLP) Call(inputs []float64) []*Value {
	x := make([]*Value, 0)
	for _, ii := range inputs {
		x = append(x, NewValue(ii, ""))
	}

	for _, l := range m.Layers {
		x = l.Call(x)
	}

	return x
}

func (m *MLP) Parameters() []*Value {
	out := make([]*Value, 0)
	for _, l := range m.Layers {
		out = append(out, l.Parameters()...)
	}
	return out
}

// Generates random float64 between -1 and 1.
// Not entirely sure about this.
func randomFloat() float64 {
	if r.Intn(2) < 1 { // roughly 50% change of 0 or 1
		return r.Float64()
	}

	return r.Float64() * -1.0
}
