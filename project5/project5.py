import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image
from scipy import ndimage


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    
    G = np.hypot(Ix, Iy)
    G = G / G.max()
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    # print(angle)
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                if (22.5 <=angle[i,j] and angle[i,j] < 67.5):
                    q = img[i+1, j-1] 
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <=angle[i,j] and angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <=angle[i,j] and angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] <= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    # highThreshold = img.max() * highThresholdRatio;
    # lowThreshold = highThreshold * lowThresholdRatio;

    highThreshold = highThresholdRatio;
    lowThreshold = lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.float32)
    g_nh = np.zeros((M,N), dtype=np.float32)
    g_nl = np.zeros((M,N), dtype=np.float32)

    weak = np.float32(0.2)
    strong = np.float32(1)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    g_nh[strong_i, strong_j] = strong
    g_nl[weak_i, weak_j] = weak
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, g_nh, g_nl)

def hysteresis(img, weak, strong=1):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

KID = "Kid at playground.tif"
kid_origin  = cv2.imread(KID ,cv2.IMREAD_GRAYSCALE)/255

kernel = gaussian_kernel(5,5)
kid_blurr = cv2.filter2D(kid_origin, -1, kernel)

M,alpha = sobel_filters(kid_blurr)
suppression_img  = non_max_suppression(M,alpha)

(res, strong, weak) = threshold(suppression_img,0.04,0.1)

final = hysteresis(res , 0.2, 1)


# cv2.imshow("img",kid_origin)
# cv2.imshow("kid_blurr",kid_blurr)
cv2.imshow("M",M)

alpha = (alpha+np.pi/2)/np.pi
cv2.imshow("alpha",alpha)
cv2.imshow("strong",strong)
cv2.imshow("weak",weak)
cv2.imshow("res",res)
cv2.imshow("final",final)
cv2.waitKey(0)


def save(all_img):
    i = 0
    for img in all_img:
        img = img*255
        img_ = Image.fromarray(img.astype(np.uint8))
        img_.save("img/{}.png".format(i),dpi=(200,200))
        i+=1
all_img = [M, alpha, strong, weak, res, final]
save(all_img)