package main

import "math"

func backPropagationExample4() {
	// inputs
	x1 := NewValue(2.0, "x1")
	x2 := NewValue(0.0, "x2")
	// weights
	w1 := NewValue(-3.0, "w1")
	w2 := NewValue(1.0, "w2")
	// bias
	b := NewValue(6.881373, "b")

	x1w1 := x1.Multiply(w1).SetLabel("x1w1")
	x2w2 := x2.Multiply(w2).SetLabel("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).SetLabel("x1w1 + 2w2")
	n := x1w1x2w2.Add(b).SetLabel("n")

	// reimplementaion of tanh
	e := n.Multiply(2.0).Exp().SetLabel("e")
	o := e.Subtract(1.0).Division(e.Add(1.0)).SetLabel("o")

	o.RecursiveBackward()

	drawDot(o)
}

func backPropagationExample3() {
	a := NewValue(-2.0, "a")
	b := NewValue(3.0, "b")
	d := a.Multiply(b).SetLabel("d")
	e := a.Add(b).SetLabel("e")
	f := d.Multiply(e).SetLabel("f")

	f.RecursiveBackward()

	drawDot(f)
}

func backPropagationExample2() {
	// inputs
	x1 := NewValue(2.0, "x1")
	x2 := NewValue(0.0, "x2")
	// weights
	w1 := NewValue(-3.0, "w1")
	w2 := NewValue(1.0, "w2")
	// bias; magic number to make the gradients nice
	b := NewValue(6.881373, "b")

	x1w1 := x1.Multiply(w1).SetLabel("x1w1")
	x2w2 := x2.Multiply(w2).SetLabel("x2w2")
	x1w1x2w2 := x1w1.Add(x2w2).SetLabel("x1w1 + 2w2")
	n := x1w1x2w2.Add(b).SetLabel("n")

	o := n.Tanh().SetLabel("o")

	o.RecursiveBackward()

	drawDot(o)
}

func backPropagationExample1() {
	a := NewValue(2.0, "a")
	b := NewValue(-3.0, "b")
	c := NewValue(10.0, "c")
	e := a.Multiply(b).SetLabel("e")
	d := e.Add(c).SetLabel("d")
	f := NewValue(-2.0, "f")
	g := f.Multiply(d).SetLabel("g")
	L := g.Exp().SetLabel("L")

	L.RecursiveBackward()

	drawDot(L)
}

func tanhPlot() {
	var twoD = make([][]float64, 0)
	for i := -5.0; i < 5.0; i += 0.2 {
		twoD = append(twoD, []float64{i, math.Tanh(i)})
	}
	scatterGraph(twoD)
}

func funcPlot() {
	var twoD = make([][]float64, 0)
	for i := -5.0; i < 5.0; i += 0.25 {
		twoD = append(twoD, []float64{i, f(i)})
	}
	scatterGraph(twoD)
}

func f(x float64) float64 {
	return 3*(math.Pow(x, 2)) - 4*x + 5
}
