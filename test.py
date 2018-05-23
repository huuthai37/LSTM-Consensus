import numpy as np
import time
start = time.time()
for i in range(20*3*8):
    nstack = np.random.rand(229,229,20)
    nstack/=255
    # nstack_nor = nstack - nstack.mean(axis=2, keepdims=True)
    nstack_nor = nstack - nstack.mean()
print (time.time() - start)