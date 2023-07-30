import numpy as np


# Whatever this is, it's utterly cursed.
def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


# I have absolutely no idea how this works and I don't like that.
def uniform(step, steps, n, context_size, strides, overlap, closed_loop=True):
    if n <= context_size:
        yield list(range(n))
        return
    strides = min(strides, int(np.ceil(np.log2(n / context_size))) + 1)
    for stride in 1 << np.arange(strides):
        pad = int(round(n * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * stride) + pad,
            n + pad + (0 if closed_loop else -overlap),
            (context_size * stride - overlap),
        ):
            yield [e % n for e in range(j, j + context_size * stride, stride)]
