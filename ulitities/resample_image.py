

import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt


input_file = '/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/originalLabelandImages/rice/test/testlabel_2.png'
output_file ='/media/omnisky/e0331d4a-a3ea-4c31-90ab-41f5b0ee2663/originalLabelandImages/rice/test/testlabel_2_resampled.png'

s = 2

if __name__=="__main__":
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    a,b = img.shape[:2]
    m = int(a/2)
    n=int(b/2)

    #
    # result = []
    # for x in range(m):
    #     for y in range(n):
    #         t = img[s*x:s*(x+1), s*y:s*(y+1)].mean()
    #         result.append(t)
    # down_img = np.array((img), np.uint8).reshape(m,n)

    result = cv2.resize(img, (n,m))

    plt.imshow(result)
    plt.show()

    cv2.imwrite(output_file,result)
