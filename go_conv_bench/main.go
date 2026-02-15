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

// Traditional method: check each neighbour individually and apply rules.
func stepTraditional(current, next Grid) {
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			count := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx := (x + dx + size) % size
					ny := (y + dy + size) % size
					if current.get(nx, ny) {
						count++
					}
				}
			}
			if current.get(x, y) {
				next.set(x, y, count == 2 || count == 3)
			} else {
				next.set(x, y, count == 3)
			}
		}
	}
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

func display(label string, g Grid) {
	displaySize := size
	if displaySize > 80 {
		displaySize = 80
	}
	for y := 0; y < displaySize; y++ {
		if y == 0 {
			fmt.Printf("  %-*s", displaySize*2, label)
		} else {
			fmt.Printf("  ")
		}
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

func gridsEqual(a, b Grid) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func benchmark(iterations int) {
	grid := randomGrid()

	tradA := newGrid()
	tradB := newGrid()
	convA := newGrid()
	convB := newGrid()
	copy(tradA, grid)
	copy(convA, grid)

	// Benchmark traditional
	start := time.Now()
	for i := 0; i < iterations; i++ {
		stepTraditional(tradA, tradB)
		tradA, tradB = tradB, tradA
	}
	tradDur := time.Since(start)

	// Benchmark convolution (integer)
	start = time.Now()
	for i := 0; i < iterations; i++ {
		stepConvolution(convA, convB)
		convA, convB = convB, convA
	}
	convDur := time.Since(start)

	// Verify they produce the same result
	match := "MATCH"
	if !gridsEqual(tradA, convA) {
		match = "MISMATCH"
	}

	fmt.Printf("Benchmark: %d iterations on %dx%d grid (integer kernel)\n\n", iterations, size, size)
	fmt.Printf("  Traditional:      %v  (%v/iter)\n", tradDur, tradDur/time.Duration(iterations))
	fmt.Printf("  Convolution(int): %v  (%v/iter)\n", convDur, convDur/time.Duration(iterations))
	fmt.Println()

	fmt.Printf("  Result after %d iterations: %s\n", iterations, match)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: conv_int <size> [bench <iterations>]")
		fmt.Println("  conv_int 100              - visual side-by-side comparison on 100x100 grid")
		fmt.Println("  conv_int 100 bench 10000  - benchmark 10000 iterations on 100x100 grid")
		return
	}

	s, err := strconv.Atoi(os.Args[1])
	if err != nil || s <= 0 {
		fmt.Println("Error: size must be a positive integer")
		return
	}
	size = s

	if len(os.Args) > 2 && os.Args[2] == "bench" {
		iterations := 100000
		if len(os.Args) > 3 {
			if n, err := strconv.Atoi(os.Args[3]); err == nil && n > 0 {
				iterations = n
			}
		}
		benchmark(iterations)
		return
	}

	grid := randomGrid()

	tradA := newGrid()
	tradB := newGrid()
	convA := newGrid()
	convB := newGrid()
	copy(tradA, grid)
	copy(convA, grid)

	for gen := 0; ; gen++ {
		fmt.Print("\033[H\033[2J")

		display("Traditional", tradA)
		fmt.Println()
		display("Convolution(int)", convA)

		fmt.Printf("Generation %d  (%dx%d)", gen, size, size)
		if gridsEqual(tradA, convA) {
			fmt.Println("  [MATCH]")
		} else {
			fmt.Println("  [MISMATCH!]")
		}

		time.Sleep(100 * time.Millisecond)
		stepTraditional(tradA, tradB)
		tradA, tradB = tradB, tradA
		stepConvolution(convA, convB)
		convA, convB = convB, convA
	}
}
