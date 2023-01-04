import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image
from random import randint
import math
import random

img_path = "Kid2 degraded.tiff"
# img_path = "kid.tif"
img_ori = cv2.imread(img_path,0)

def filter_process(img, alpha):
    vector1 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vector1.append(img[i, j])
    vector1.sort()
    sum = 0
    for i in range(math.floor(alpha/2.), len(vector1) - math.floor(alpha/2.)):
        sum += vector1[i]
    sum /= len(vector1) - 2 * math.floor(alpha/2.)
    return int(sum)

kernel_size = 5
alpha = 16

plt.subplot(121), plt.imshow(img_ori, cmap='gray')
side_leng = math.floor(kernel_size/2.)
height, width = img_ori.shape
padded_img = np.zeros((height+2*side_leng, width+2*side_leng))
padded_img[2:height+2, 2:width+2] = img_ori
de_noise = np.zeros((height, width))
side_leng = math.floor(kernel_size/2.)
for i in range(side_leng, height+side_leng):
    for j in range(side_leng, width+side_leng):
        tmp_img = padded_img[i - side_leng:i + side_leng + 1, j - side_leng:j + side_leng + 1]
        de_noise[i-2, j-2] = filter_process(tmp_img, alpha)

# plt.subplot(122),plt.imshow(de_noise, cmap='gray')
# plt.show()

# hist_de_noise = cv2.calcHist([de_noise.astype(np.uint8)], [0], None, [256], [0, 256])
# hist_img_ori = cv2.calcHist([img_ori], [0], None, [256], [0, 256])
# print(type(hist_img_ori))
# plt.title("Histogram")
# plt.xlabel("gray levels")
# plt.ylabel("pixel numbers")
# nois_p = (hist_img_ori-hist_de_noise)/(800*800)
# plt.plot(nois_p)
# print("Pa of noise model = ",nois_p[0])
# print("Pb of noise model = ",nois_p[255])
# plt.show()


img_use = cv2.copyMakeBorder(de_noise, 0, 800, 0, 800, cv2.BORDER_CONSTANT)
# img_use = de_noise
g = np.fft.fft2(img_use)
G = np.fft.fftshift(g)

D0 = 250
n = 10
M = img_use.shape[0]
N = img_use.shape[1]
B = 0.414
BLPF = np.zeros((M,N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        BLPF[u,v] = 1 / (1 + B*(D/D0)**(2*n))

d0 = [100, 150, 200, 250]
final = []
GLPF = np.zeros((M,N), dtype=np.float32)
for D0 in d0:
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2) 
            GLPF[u,v] = np.exp(-D**2/(2*D0*D0))
    F = G/GLPF*BLPF
    lf = np.abs(np.fft.ifft2(F))
    lf = lf[0:800,0:800]
    deblurr = Image.fromarray(lf)
    deblurr = deblurr.convert('RGB')
    deblurr.save("img/deblurr{}.png".format(D0),dpi=(200,200))
    final.append(lf)
plt.subplot(221),plt.imshow(final[0], cmap='gray')
plt.subplot(222),plt.imshow(final[1], cmap='gray')
plt.subplot(223),plt.imshow(final[2], cmap='gray')
plt.subplot(224),plt.imshow(final[3], cmap='gray')
plt.show()


de_noise_model = Image.fromarray(de_noise.astype(np.uint8))
de_noise_model.save("img/de_noise_model.png",dpi=(200,200))

# output = np.zeros(img_ori.shape,np.uint8)
# prob = 0.8
# thres = 1 - prob
# for i in range(img_ori.shape[0]):
#         for j in range(img_ori.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = 0
# plt.imshow(output, cmap='gray')
# plt.show()