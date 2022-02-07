import cv2
import numpy as np
import itertools


fname_common = "/home/megu/ECoG_CNNs/Result/2022-02-04_frequency_KT"

fname_Anesthetized_smoothgrad_mean = fname_common + '/Anesthetized/smooth_grad/mean.png'

a= np.arange(27).reshape(3, 3, 3)
print(a)




img_Anesthetized_smoothgrad_mean = cv2.imread(fname_Anesthetized_smoothgrad_mean)
img_resize = cv2.resize(img_Anesthetized_smoothgrad_mean, dsize=(200, 200))

'''
for i, j in itertools.product(range(200), range(200)):
    print(img_resize[i][j])
    if not img_resize[i][j][0]== img_resize[i][j][1] == img_resize[i][j][2]:
        print("diff")
        break

'''


#print(img_Anesthetized_smoothgrad_mean.shape)
#print(img_resize.shape)

img_2div = np.mean(img_resize, axis=2)
print(img_2div)
line_mean = np.mean(img_2div, axis=1)
print(line_mean)
print(type(line_mean))
#cv2.imwrite(fname_common + "/ane_mean.png", img_mean)