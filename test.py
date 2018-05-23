import numpy as np
import time
import cv2

start = time.time()
for i in range(3*8):
    nstack = np.zeros((256,256,20))
    img = np.random.rand(256,256,20)

    for i in range(10):
        nstack[:,:,2*i] = img[:,:,2*i]
        nstack[:,:,2*i+1] = img[:,:,2*i+1]

    nstack = cv2.resize(nstack, (299, 299))
    nstack = nstack.astype('float16',copy=False)
    nstack/=255
    nstack_nor = nstack - nstack.mean()
print (time.time() - start)