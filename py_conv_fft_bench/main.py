import sys
import time
import numpy as np
from numpy.fft import fft2, ifft2

size = 0


def new_grid():
    return np.zeros(size * size, dtype=bool)


def random_grid():
    return np.random.randint(0, 2, size * size, dtype=bool)


def step_traditional(current, next_grid):
    grid = current.reshape(size, size).astype(np.int8)
    count = np.zeros((size, size), dtype=np.int8)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            count += np.roll(np.roll(grid, -dy, axis=0), -dx, axis=1)
    alive = grid.astype(bool)
    result = (alive & ((count == 2) | (count == 3))) | (~alive & (count == 3))
    next_grid[:] = result.ravel()


kernel_fft: np.ndarray = None  # type: ignore[assignment]


def init_fft():
    global kernel_fft
    k = np.array([
        [2.0, 2.0, 2.0],
        [2.0, 1.0, 2.0],
        [2.0, 2.0, 2.0],
    ])
    kernel = np.zeros((size, size))
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            fy = (dy + size) % size
            fx = (dx + size) % size
            kernel[fy, fx] = k[dy + 1][dx + 1]
    kernel_fft = fft2(kernel)


def step_fft(current, next_grid):
    data = current.reshape(size, size).astype(np.float64)
    result = np.real(ifft2(fft2(data) * kernel_fft))
    # accounting for slight inaccuracies in the floats and fft
    # smallest steps of outputs are 1 so comparisons sit halfway
    flat_next = (result >= 4.5) & (result <= 7.5)
    next_grid[:] = flat_next.ravel()


def grids_equal(a, b):
    return np.array_equal(a, b)


def benchmark(iterations):
    init_fft()
    grid = random_grid()

    print(f"Grid: {size}x{size}, FFT padded: {size}x{size}")
    print(f"Running {iterations} iterations...\n")

    trad_a = grid.copy()
    trad_b = new_grid()
    fft_a = grid.copy()
    fft_b = new_grid()

    trad_dur = 0.0
    fft_dur = 0.0

    for i in range(iterations):
        start = time.perf_counter()
        step_traditional(trad_a, trad_b)
        trad_dur += time.perf_counter() - start
        trad_a, trad_b = trad_b, trad_a

        start = time.perf_counter()
        step_fft(fft_a, fft_b)
        fft_dur += time.perf_counter() - start
        fft_a, fft_b = fft_b, fft_a

        mismatches = int(np.sum(trad_a != fft_a))
        if mismatches > 0:
            for y in range(size):
                for x in range(size):
                    if trad_a[y * size + x] != fft_a[y * size + x]:
                        print(f"  Iter {i}: mismatch at ({x},{y}) trad={trad_a[y*size+x]} fft={fft_a[y*size+x]}")
            print(f"  Iter {i}: {mismatches} total mismatches")

    match = "MATCH" if grids_equal(trad_a, fft_a) else "MISMATCH"

    trad_per_iter = trad_dur / iterations * 1000
    fft_per_iter = fft_dur / iterations * 1000

    print(f"  Traditional:  {trad_dur:.6f}s  ({trad_per_iter:.3f} ms/iter)")
    print(f"  FFT Conv:     {fft_dur:.6f}s  ({fft_per_iter:.3f} ms/iter)")
    print()
    print(f"  Result after {iterations} iterations: {match}")


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def main():
    global size

    if len(sys.argv) < 2:
        print("Usage: main.py <size> [bench <iterations>]")
        print("  Size must be a power of 2 (64, 128, 256, 512, 1024, ...)")
        print("  main.py 128           - run visual simulation on 128x128 grid")
        print("  main.py 1024 bench 50 - benchmark 50 iterations on 1024x1024 grid")
        return

    try:
        s = int(sys.argv[1])
        assert s > 0
    except (ValueError, AssertionError):
        print("Error: size must be a positive integer")
        return

    if s & (s - 1) != 0:
        nxt = next_pow2(s)
        answer = input(f"Size must be a power of 2. The next power of 2 is {nxt}. Use that? [Y/n] ")
        if answer and answer[0] not in ("y", "Y"):
            print("Aborted.")
            return
        s = nxt

    size = s

    if len(sys.argv) > 2 and sys.argv[2] == "bench":
        iterations = 100
        if len(sys.argv) > 3:
            try:
                n = int(sys.argv[3])
                if n > 0:
                    iterations = n
            except ValueError:
                pass
        benchmark(iterations)
        return

    init_fft()
    grid = random_grid()

    trad_a = grid.copy()
    trad_b = new_grid()
    fft_a = grid.copy()
    fft_b = new_grid()

    gen = 0
    while True:
        print("\033[H\033[2J", end="")
        status = "MATCH" if grids_equal(trad_a, fft_a) else "MISMATCH!"
        print(f"Generation {gen}  ({size}x{size})  [{status}]")
        print()

        time.sleep(0.1)
        step_traditional(trad_a, trad_b)
        trad_a, trad_b = trad_b, trad_a
        step_fft(fft_a, fft_b)
        fft_a, fft_b = fft_b, fft_a
        gen += 1


if __name__ == "__main__":
    main()
