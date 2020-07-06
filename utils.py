import sys
from tqdm import tqdm
from tqdm import trange


def xrange(iters, prefix=None, Epoch=None, **kwargs):
    if Epoch is not None and prefix is None:
        prefix = "Epoch " + str(Epoch)
    return trange(
        int(iters),
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
        desc=prefix,
        **kwargs
    )


def range_iterator(iters, prefix=None, Epoch=None, **kwargs):
    if Epoch is not None and prefix is None:
        prefix = "Epoch " + str(Epoch)
    return tqdm(
        iters, file=sys.stdout, leave=False, dynamic_ncols=True, desc=prefix, **kwargs
    )
