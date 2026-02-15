package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var size int

type Grid []bool

var kernel = [3][3]int{
	{2, 2, 2},
	{2, 1, 2},
	{2, 2, 2},
}

func newGrid() Grid {
	return make(Grid, size*size)
}

func (g Grid) get(x, y int) bool {
	return g[y*size+x]
}

func (g Grid) set(x, y int, v bool) {
	g[y*size+x] = v
}

func randomGrid() Grid {
	g := newGrid()
	for i := range g {
		g[i] = rand.Intn(2) == 1
	}
	return g
}

// Convolution method: convolve grid with integer kernel, then threshold.
func stepConvolution(current, next Grid) {
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			sum := 0
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					ny := (y + ky + size) % size
					nx := (x + kx + size) % size
					if current.get(nx, ny) {
						sum += kernel[ky+1][kx+1]
					}
				}
			}
			next.set(x, y, sum >= 5 && sum <= 7)
		}
	}
}

func display(g Grid) {
	displaySize := size
	if displaySize > 80 {
		displaySize = 80
	}
	for y := 0; y < displaySize; y++ {
		for x := 0; x < displaySize; x++ {
			if g.get(x, y) {
				fmt.Print("██")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: conv <size>")
		fmt.Println("  conv 100  - visual simulation on 100x100 grid")
		return
	}

	s, err := strconv.Atoi(os.Args[1])
	if err != nil || s <= 0 {
		fmt.Println("Error: size must be a positive integer")
		return
	}
	size = s

	a := randomGrid()
	b := newGrid()

	var calcDur, renderDur time.Duration
	for gen := 0; ; gen++ {
		start := time.Now()
		fmt.Print("\033[H\033[2J")
		display(a)
		fmt.Printf("Generation %d  (%dx%d)  calc: %v  render: %v\n", gen, size, size, calcDur, renderDur)
		renderDur = time.Since(start)

		time.Sleep(100 * time.Millisecond)
		start = time.Now()
		stepConvolution(a, b)
		calcDur = time.Since(start)
		a, b = b, a
	}
}
