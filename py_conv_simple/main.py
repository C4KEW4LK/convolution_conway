import numpy as np
from scipy.signal import convolve2d
import os
import sys
import time

WIDTH = 20
HEIGHT = 20

KERNEL = np.array([
    [2, 2, 2],
    [2, 1, 2],
    [2, 2, 2],
], dtype=np.int32)


def random_grid():
    return np.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=np.int32)


def step_convolution_lib(current):
    conv = convolve2d(current, KERNEL, mode="same", boundary="wrap")
    return ((conv >= 5) & (conv <= 7)).astype(np.int32)


def step_convolution_manual(current):
    next_grid = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            s = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    ny = (y + ky) % HEIGHT
                    nx = (x + kx) % WIDTH
                    if current[ny][nx]:
                        s += KERNEL[ky + 1][kx + 1]
            next_grid[y][x] = 5 <= s <= 7
    return next_grid


def display(grid):
    for row in grid:
        print("  ", end="")
        for cell in row:
            print("██" if cell else "  ", end="")
        print()


def main():
    use_lib = "--lib" in sys.argv
    use_manual = "--manual" in sys.argv

    if not use_lib and not use_manual:
        use_lib = True

    step = step_convolution_lib if use_lib else step_convolution_manual
    label = "scipy" if use_lib else "manual"

    grid = random_grid()
    gen = 0
    iter_ms = 0.0
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Generation {gen}  [{label}]\n")
        display(grid)
        print(f"\n  {iter_ms:.3f} ms/iter")
        start = time.perf_counter()
        grid = step(grid)
        iter_ms = (time.perf_counter() - start) * 1000
        time.sleep(0.1)
        gen += 1


if __name__ == "__main__":
    main()
