package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"time"

	"gonum.org/v1/gonum/dsp/fourier"
)

var size int // grid is size x size

func nextPow2(n int) int {
	return int(math.Pow(2, math.Ceil(math.Log2(float64(n)))))
}

var (
	fftN int // padded FFT dimension (square)
)

// Grid stored as flat slice, row-major: grid[y*size+x]
type Grid []bool

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

// --- 2D FFT convolution using gonum/dsp/fourier ---
// Following the gonum 2D FFT pattern: real FFT on rows, complex FFT on columns.
// Only c/2+1 coefficients are stored per row (exploiting real-input symmetry).

var (
	realFFT    *fourier.FFT
	cmplxFFT   *fourier.CmplxFFT
	kernelFreq []complex128 // pre-transformed kernel: fftN rows × halfC cols
	freqBuf    []complex128 // work buffer in freq domain: fftN × halfC
	colBuf     []complex128 // column scratch buffer (length fftN)
	realBuf    []float64    // row scratch buffer (length fftN)
	halfC      int          // fftN/2 + 1 reduced column count
	normInv    float64      // 1/(fftN*fftN) for unnormalized inverse
)

func initFFT() {
	fftN = size
	halfC = fftN/2 + 1
	realFFT = fourier.NewFFT(fftN)
	cmplxFFT = fourier.NewCmplxFFT(fftN)
	normInv = 1.0 / float64(fftN*fftN)

	freqBuf = make([]complex128, fftN*halfC)
	colBuf = make([]complex128, fftN)
	realBuf = make([]float64, fftN)
	kernelFreq = make([]complex128, fftN*halfC)

	// Build kernel in spatial domain
	kernelReal := make([]float64, fftN*fftN)
	k := [3][3]float64{
		{2, 2, 2},
		{2, 1, 2},
		{2, 2, 2},
	}
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			fy := (dy + fftN) % fftN
			fx := (dx + fftN) % fftN
			kernelReal[fy*fftN+fx] = k[dy+1][dx+1]
		}
	}

	// Forward 2D FFT of kernel
	// First axis: real FFT on each row → halfC complex coefficients
	for y := 0; y < fftN; y++ {
		realFFT.Coefficients(kernelFreq[y*halfC:(y+1)*halfC], kernelReal[y*fftN:(y+1)*fftN])
	}
	// Second axis: complex FFT on each column
	for x := 0; x < halfC; x++ {
		for y := 0; y < fftN; y++ {
			colBuf[y] = kernelFreq[y*halfC+x]
		}
		cmplxFFT.Coefficients(colBuf, colBuf)
		for y := 0; y < fftN; y++ {
			kernelFreq[y*halfC+x] = colBuf[y]
		}
	}
}

func stepFFT(current, next Grid) {
	// Forward 2D FFT of grid
	// First axis: real FFT on each row
	for y := 0; y < fftN; y++ {
		for x := 0; x < fftN; x++ {
			if current.get(x, y) {
				realBuf[x] = 1
			} else {
				realBuf[x] = 0
			}
		}
		realFFT.Coefficients(freqBuf[y*halfC:(y+1)*halfC], realBuf)
	}
	// Second axis: complex FFT on each column
	for x := 0; x < halfC; x++ {
		for y := 0; y < fftN; y++ {
			colBuf[y] = freqBuf[y*halfC+x]
		}
		cmplxFFT.Coefficients(colBuf, colBuf)
		for y := 0; y < fftN; y++ {
			freqBuf[y*halfC+x] = colBuf[y]
		}
	}

	// Pointwise multiply in frequency domain
	for i := range freqBuf {
		freqBuf[i] *= kernelFreq[i]
	}

	// Inverse 2D FFT
	// Second axis inverse: complex IFFT on each column
	for x := 0; x < halfC; x++ {
		for y := 0; y < fftN; y++ {
			colBuf[y] = freqBuf[y*halfC+x]
		}
		cmplxFFT.Sequence(colBuf, colBuf)
		for y := 0; y < fftN; y++ {
			freqBuf[y*halfC+x] = colBuf[y]
		}
	}
	// First axis inverse: real IFFT on each row, normalize, and apply rules
	for y := 0; y < fftN; y++ {
		realFFT.Sequence(realBuf, freqBuf[y*halfC:(y+1)*halfC])
		for x := 0; x < size; x++ {
			v := realBuf[x] * normInv
			next.set(x, y, v >= 4.5 && v <= 7.5)
		}
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
	initFFT()
	grid := randomGrid()

	fmt.Printf("Grid: %dx%d, FFT padded: %dx%d\n", size, size, fftN, fftN)
	fmt.Printf("Running %d iterations...\n\n", iterations)

	tradA := newGrid()
	tradB := newGrid()
	fftA := newGrid()
	fftB := newGrid()
	copy(tradA, grid)
	copy(fftA, grid)

	var tradDur, fftDur time.Duration

	for i := 0; i < iterations; i++ {
		start := time.Now()
		stepTraditional(tradA, tradB)
		tradDur += time.Since(start)
		tradA, tradB = tradB, tradA

		start = time.Now()
		stepFFT(fftA, fftB)
		fftDur += time.Since(start)
		fftA, fftB = fftB, fftA

		mismatches := 0
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				if tradA.get(x, y) != fftA.get(x, y) {
					fmt.Printf("  Iter %d: mismatch at (%d,%d) trad=%v fft=%v\n", i, x, y, tradA.get(x, y), fftA.get(x, y))
					mismatches++
				}
			}
		}
		if mismatches > 0 {
			fmt.Printf("  Iter %d: %d total mismatches\n", i, mismatches)
		}
	}
	runtime.KeepAlive(tradA)
	runtime.KeepAlive(fftA)

	match := "MATCH"
	if !gridsEqual(tradA, fftA) {
		match = "MISMATCH"
	}

	tradPerIter := float64(tradDur.Nanoseconds()) / float64(iterations)
	fftPerIter := float64(fftDur.Nanoseconds()) / float64(iterations)

	fmt.Printf("  Traditional:  %v  (%.3f ms/iter)\n", tradDur, tradPerIter/1e6)
	fmt.Printf("  FFT Conv:     %v  (%.3f ms/iter)\n", fftDur, fftPerIter/1e6)
	fmt.Println()

	fmt.Printf("  Result after %d iterations: %s\n", iterations, match)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: conv_fft <size> [bench <iterations>]")
		fmt.Println("  Size must be a power of 2 (64, 128, 256, 512, 1024, ...)")
		fmt.Println("  conv_fft 128           - run visual simulation on 128x128 grid")
		fmt.Println("  conv_fft 1024 bench 50 - benchmark 50 iterations on 1024x1024 grid")
		return
	}

	s, err := strconv.Atoi(os.Args[1])
	if err != nil || s <= 0 {
		fmt.Println("Error: size must be a positive integer")
		return
	}
	if s&(s-1) != 0 {
		next := nextPow2(s)
		fmt.Printf("Size must be a power of 2. The next power of 2 is %d. Use that? [Y/n] ", next)
		var answer string
		fmt.Scanln(&answer)
		if answer != "" && answer[0] != 'y' && answer[0] != 'Y' {
			fmt.Println("Aborted.")
			return
		}
		s = next
	}
	size = s

	if len(os.Args) > 2 && os.Args[2] == "bench" {
		iterations := 100
		if len(os.Args) > 3 {
			if n, err := strconv.Atoi(os.Args[3]); err == nil && n > 0 {
				iterations = n
			}
		}
		benchmark(iterations)
		return
	}

	initFFT()
	grid := randomGrid()

	tradA := newGrid()
	tradB := newGrid()
	fftA := newGrid()
	fftB := newGrid()
	copy(tradA, grid)
	copy(fftA, grid)

	// Cap display to fit a reasonable terminal (each cell = 2 chars wide)
	displaySize := size
	if displaySize > 80 {
		displaySize = 80
	}

	for gen := 0; ; gen++ {
		fmt.Print("\033[H\033[2J")
		for y := 0; y < displaySize; y++ {
			for x := 0; x < displaySize; x++ {
				if fftA.get(x, y) {
					fmt.Print("██")
				} else {
					fmt.Print("  ")
				}
			}
			fmt.Println()
		}
		fmt.Printf("Generation %d  (%dx%d)", gen, size, size)
		if gridsEqual(tradA, fftA) {
			fmt.Println("  [MATCH]")
		} else {
			fmt.Println("  [MISMATCH!]")
		}

		time.Sleep(100 * time.Millisecond)
		stepTraditional(tradA, tradB)
		tradA, tradB = tradB, tradA
		stepFFT(fftA, fftB)
		fftA, fftB = fftB, fftA
	}
}
