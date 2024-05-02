
import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
from functools import partial

import tqdm
from itertools import islice

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


num_classes_per_batch = 10
num_classes = 1000
num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)

def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

classnames=IMAGENET_CLASSNAMES

for batch in iter_wrap(batched(classnames, num_classes_per_batch)):
    print(batch)
    assert False
