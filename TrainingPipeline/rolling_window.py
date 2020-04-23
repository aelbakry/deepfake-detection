from itertools import islice
import numpy as np

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


x = [[0, 1, 2, 3, 4, 5, 6], [00, 11, 22, 33, 44, 55, 66]]

x_rolled = []

for i in range(len(x)):
        rolled = window(x[i], n=5)
        x_rolled.append(list(rolled))

print(x_rolled)
print(np.shape(x_rolled))
