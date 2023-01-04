import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image

flower = "LovePeace rose.tif"
bgr = cv2.imread(flower)

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
b = bgr[:,:,0]
g = bgr[:,:,1]
r = bgr[:,:,2]
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

plt.subplot(221),plt.imshow(rgb)
plt.subplot(222),plt.imshow(b, cmap='gray')
plt.subplot(223),plt.imshow(g, cmap='gray')
plt.subplot(224),plt.imshow(r, cmap='gray')
plt.show()

plt.subplot(221),plt.imshow(hsv)
plt.subplot(222),plt.imshow(h, cmap='gray')
plt.subplot(223),plt.imshow(s, cmap='gray')
plt.subplot(224),plt.imshow(v, cmap='gray')
plt.show()
all_img = [r,g,b,h,s,v]
img_name = ['r','g','b','h','s','v']
i = 0
for img in all_img:
    img_ = Image.fromarray(img)
    img_.save("img/{}.png".format(img_name[i]),dpi=(200,200))
    i+=1

kernel = np.array([[-1, -1, -1],
                   [-1, 9,-1],
                   [-1, -1, -1]])
bgr_sharp = cv2.filter2D(src=bgr, ddepth=-1, kernel=kernel)
v_sharp = cv2.filter2D(src=v, ddepth=-1, kernel=kernel)
hsv_sharp = np.dstack((h,s,v_sharp))
after_hsv2bgr = cv2.cvtColor(hsv_sharp, cv2.COLOR_HSV2BGR)
cv2.imshow("img",bgr_sharp)
cv2.waitKey(0)
cv2.imshow("img0",after_hsv2bgr)
cv2.waitKey(0)
diff = cv2.subtract(bgr_sharp, after_hsv2bgr)
cv2.imshow("img1",diff)
cv2.waitKey(0)

all_img = [bgr_sharp,after_hsv2bgr,diff]
i = 0
for img in all_img:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(img)
    # img_  = img_ .convert('RGB')
    img_.save("img/{}.png".format(i),dpi=(200,200))
    i+=1



