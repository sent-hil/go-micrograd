package main

import (
	"bytes"
	"fmt"
	"log"
	"math/rand"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func drawDot(root *Value) {
	g := graphviz.New()
	graph, err := g.Graph()
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatal(err)
		}
		g.Close()
	}()

	if _, err := recursiveDraw(graph, root); err != nil {
		log.Fatal(err)
	}

	var buf bytes.Buffer
	if err := g.Render(graph, "dot", &buf); err != nil {
		log.Fatal(err)
	}
	if err := g.RenderFilename(graph, graphviz.PNG, "./graph.png"); err != nil {
		log.Fatal(err)
	}
}

func recursiveDraw(graph *cgraph.Graph, v *Value) (*cgraph.Node, error) {
	node, err := graph.CreateNode(fmt.Sprintf("%s | data %.4f | grad %.4f", v.Label, v.Data, v.Gradient))
	if err != nil || v.Operator == "" {
		return node, err
	}

	// graphviz treats Node with same name as same node, so we append random
	// number to make it an unique node
	operatorNode, err := graph.CreateNode(fmt.Sprintf("%s (%1d)", v.Operator, rand.Intn(9999)))
	if err != nil {
		return node, err
	}

	if _, err = graph.CreateEdge("", node, operatorNode); err != nil {
		return node, err
	}

	for _, c := range v.Children {
		childNode, err := recursiveDraw(graph, c)
		if err != nil {
			return node, err
		}

		if _, err = graph.CreateEdge("", operatorNode, childNode); err != nil {
			return node, err
		}
	}

	return node, nil
}

func scatterGraph(twoD [][]float64) {
	var pts = make(plotter.XYs, len(twoD))
	for i := range pts {
		pts[i].X = twoD[i][0]
		pts[i].Y = twoD[i][1]
	}

	p := plot.New()
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err := plotutil.AddLinePoints(p, "", pts)
	if err != nil {
		log.Fatal(err)
	}

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "plot.png"); err != nil {
		log.Fatal(err)
	}
}
