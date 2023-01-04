import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image
from random import randint

path1 = "kid.tif"
path2 = "fruit.tif"

def work(img_path):
    img = cv2.imread(img_path,0)

    #Fourier magnitude spectra 600*600
    save_f = np.fft.fft2(img)
    save_F = np.fft.fftshift(save_f)
    print(save_F)
    magnitude_spectrum = 20*np.log(np.abs(save_F))
    #Fourier magnitude spectra 1200*1200
    img = cv2.copyMakeBorder(img, 0, 600, 0, 600, cv2.BORDER_CONSTANT)
    f = np.fft.fft2(img)
    F = np.fft.fftshift(f)

    tables = []
    for i in range(0,300):
        for j in range(0,600):
            tables.append([magnitude_spectrum[i][j],i,j])
    tables = sorted(tables)
    
    print("SMALL to BIG 25 DFT frequencies of {}:".format(img_path[:-4]))
    for i in range(-25,0):
        print(tables[i][1:3])


    M,N = (1200,1200)
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 200
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))
    #through lowpass
    LG=F*H
    abs_L = np.abs(H)
    LG= np.fft.ifftshift(LG)
    lg=np.abs(np.fft.ifft2(LG))
    #through highpass
    HG=F*(1-H)
    abs_H = np.abs(1-H)
    HG= np.fft.ifftshift(HG)
    hg=np.abs(np.fft.ifft2(HG))

    plt.subplot(221),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(magnitude_spectrum , cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.title('High Pass'),plt.imshow(hg[0:600,0:600], cmap='gray'),plt.axis('off')
    plt.subplot(224),plt.title('Low pass'),plt.imshow(lg[0:600,0:600], cmap='gray'),plt.axis('off')
    # plt.subplot(223),plt.title('High Pass'),plt.imshow(abs_H, cmap='gray'),plt.axis('off')
    # plt.subplot(224),plt.title('Low pass'),plt.imshow(abs_L, cmap='gray'),plt.axis('off')
    plt.axis('off')
    plt.show()

    output_magnitude_spectrum=Image.fromarray(magnitude_spectrum)
    output_magnitude_spectrum=output_magnitude_spectrum.convert('RGB')
    output_magnitude_spectrum.save("img/{}_magnitude_spectrum.png".format(img_path[:-4]),dpi=(150,150))
    
    output_abs_H = Image.fromarray(abs_H*255)
    output_abs_H = output_abs_H.convert('RGB')
    output_abs_H.save("img/output_abs_H.png",dpi=(150,150))

    output_abs_L = Image.fromarray(abs_L*255)
    output_abs_L = output_abs_L.convert('RGB')
    output_abs_L.save("img/output_abs_L.png",dpi=(150,150))

    output_abs_H_img = Image.fromarray(hg[0:600,0:600])
    output_abs_H_img = output_abs_H_img.convert('RGB')
    output_abs_H_img.save("img/{}_output_abs_H_img.png".format(img_path[:-4]),dpi=(150,150))

    output_abs_L_img = Image.fromarray(lg[0:600,0:600])
    output_abs_L_img = output_abs_L_img.convert('RGB')
    output_abs_L_img.save("img/{}_output_abs_L_img.png".format(img_path[:-4]),dpi=(150,150))

work(path1)
work(path2)