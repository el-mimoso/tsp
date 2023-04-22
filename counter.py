from collections import Counter
import numpy as np
from res.d18v3 import *

count = Counter(progress)
items = count.keys()
values = count.values()
print(values)
lVals = list(values)
print(np.max(lVals))
print(sorted(lVals))

# print("Frec: "+ values)