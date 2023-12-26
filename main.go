package main

import (
	"fmt"
)

func main() {
	// These will output plot.png to root folder, uncomment them 1 at a time.
	//funcPlot()
	//tanhPlot()

	// These will output graph.png to root folder, uncomment them 1 at a time.
	backPropagationExample1()
	//backPropagationExample2()
	//backPropagationExample3()
	//backPropagationExample4()

	fmt.Println()

	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	n := NewMLP(3, []int{4, 4, 1})

	for i := 0; i < 20; i++ {
		loss := NewValue(0.0, "loss")
		for i, x := range xs {
			ypred := n.Call(x)[0]
			fmt.Printf("Expected:%.2f, Got:%.2f\n", ys[i], ypred.Data)

			loss = loss.Add(ypred.Subtract(ys[i]).Power(2))
		}

		for _, p := range n.Parameters() {
			p.Gradient = 0.0
		}

		loss.RecursiveBackward()

		fmt.Printf("Iteration:%d Loss:%.2f\n", i, loss.Data)

		for _, p := range n.Parameters() {
			p.Data += -0.05 * p.Gradient
		}

		fmt.Println()
	}
}
