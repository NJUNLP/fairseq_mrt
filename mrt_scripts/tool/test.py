import numpy as np
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = [a, b]
c = np.sum(c, axis=0).tolist()
print(c)
# python3 mrt_scripts/tool/test.py