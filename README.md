# conv_conway

Conway's Game of Life implemented using convlution compared to traditional neighbor-counting, in both Go and Python. There is no speedup vs tradional version as convolution is overall slower (addition vs multiplication) in these implementations. Just fun and interesting that it can be done.

All implementations use wrap-around boundary conditions. The bench variants run both traditional and convolution methods side-by-side, verifying that they produce identical output each iteration to prove correctness of the convolution-based approaches.

## Project Structure

```
conv_conway/
├── go_conv_simple/        Go: convolution-only simulation
├── go_conv_bench/         Go: traditional vs integer convolution benchmark
├── go_conv_fft_bench/     Go: traditional vs FFT convolution benchmark
├── py_conv_simple/        Python: convolution-only simulation (scipy / manual)
└── py_conv_fft_bench/     Python: traditional vs FFT convolution benchmark
```

## Algorithm Variants

### Traditional (`stepTraditional`)
Counts each of the 8 neighbors individually and applies the standard Conway's Game of Life rules:

1. Any live cell with fewer than 2 live neighbors dies (underpopulation).
2. Any live cell with 2 or 3 live neighbors survives to the next generation.
3. Any live cell with more than 3 live neighbors dies (overpopulation).
4. Any dead cell with exactly 3 live neighbors becomes alive (reproduction).

### Integer Convolution (`stepConvolution`)
Convolves the grid with a weighted kernel:

```
┌───┬───┬───┐
│ 2 │ 2 │ 2 │
├───┼───┼───┤
│ 2 │ 1 │ 2 │
├───┼───┼───┤
│ 2 │ 2 │ 2 │
└───┴───┴───┘
```

Neighbors are weighted 2, the center cell is weighted 1. The convolution sum encodes both the neighbor count and the cell's own state, so a single threshold range (5-7) replaces the separate alive/dead rule branches:

| Traditional rule                       | State | Neighbors | Sum (2*neighbors + cell) | Next step, Alive?  |
|----------------------------------------|-------|-----------|--------------------------|---------|
| Dead + 2 neighbors = stays dead        | Dead  | 2         | 2*2 + 0 = 4             | ❌      |
| **Dead + 3 neighbors = born**          | Dead  | 3         | **2*3 + 0 = 6**         | ✅      |
| Dead + 4 neighbors = stays dead        | Dead  | 4         | 2*4 + 0 = 8             | ❌      |
| Alive + 1 neighbor = dies              | Alive | 1         | 2*1 + 1 = 3             | ❌      |
| **Alive + 2 neighbors = survives**     | Alive | 2         | **2*2 + 1 = 5**         | ✅      |
| **Alive + 3 neighbors = survives**     | Alive | 3         | **2*3 + 1 = 7**         | ✅      |
| Alive + 4 neighbors = dies             | Alive | 4         | 2*4 + 1 = 9             | ❌      |

Only sums of 5, 6, and 7 produce a live cell — matching the traditional rules exactly.

### FFT Convolution (`stepFFT` / `step_fft`)
Uses the same kernel as the integer convolution. Pre-computes the FFT of the kernel once, then each step forward-FFTs the grid, multiplies element-wise with the kernel FFT, and inverse-FFTs back. The result is thresholded at 4.5-7.5 (halfway between the discrete sums) to give room for floating-point imprecision from the FFT. Efficient for large grids.

---

## go_conv_simple

Convolution-only Game of Life on a variable-size square grid. Generates a random grid, then runs the convolution step in a loop, rendering each generation to the terminal with Unicode block characters.

```
go run main.go 100   # visual simulation on 100x100 grid
```

## go_conv_bench

Runs traditional and integer convolution side-by-side on a variable-size square grid. Both methods start from the same random seed and step in lockstep. Each iteration the results are compared to verify the convolution approach produces identical output to the traditional rules. In visual mode it renders both grids; in bench mode it reports per-iteration timing and match status.

```
go run main.go 100              # visual side-by-side comparison on 100x100 grid
go run main.go 100 bench 10000  # benchmark 10000 iterations on 100x100 grid
```

## go_conv_fft_bench

Same side-by-side correctness verification but using FFT-accelerated convolution on variable-size grids (must be power of 2). Uses [`gonum/dsp/fourier`](https://pkg.go.dev/gonum.org/v1/gonum/dsp/fourier) for the 2D FFT — `fourier.NewFFT` (real) for row transforms and `fourier.NewCmplxFFT` for column transforms, following gonum's standard 2D pattern with the reduced `n/2+1` coefficient representation. The kernel FFT is pre-computed once, then each step does a forward FFT of the grid, element-wise multiply, and inverse FFT. The grid is stored as a flat boolean slice in row-major order.

```
go run main.go 256              # visual simulation on 256x256
go run main.go 1024 bench 50   # correctness verification over 50 iterations on 1024x1024
```

## py_conv_simple

Convolution-only Game of Life on a 20x20 grid in Python. Offers two convolution backends: `scipy.signal.convolve2d` with wrap boundary mode (`--lib`, the default), or a manual implementation (`--manual`). Renders to terminal in a loop.

```
python main.py            # visual simulation (scipy)
python main.py --manual   # visual simulation (manual convolution)
```

**Dependencies:** numpy, scipy

## py_conv_fft_bench

Python port of go_conv_fft_bench. Runs FFT convolution and traditional methods side-by-side on variable-size grids, verifying 1:1 match each iteration. The traditional step uses `np.roll` to shift the grid in all 8 directions and sum neighbors, avoiding slow Python loops. The FFT step uses [`numpy.fft.fft2`/`ifft2`](https://numpy.org/doc/stable/reference/routines.fft.html). Both operate on flat numpy boolean arrays.

```
python main.py 256             # visual simulation on 256x256
python main.py 1024 bench 50   # correctness verification over 50 iterations on 1024x1024
```

**Dependencies:** numpy
