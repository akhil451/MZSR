import cv2 
import os
import numpy as np 

scale=4.0

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)

    if len(sz)==2:
        sz = sz //modulo
        # out = imgs[0:int(sz[0]), 0:int(sz[1])]
        out = cv2.resize(img,sz)
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt // modulo
        szt =[int(S) for S in szt]
        szt=tuple(szt)
        # out = imgs[0:int(szt[0]), 0:int(szt[1
        out = cv2.resize(img,szt)

    return out

base ="/home/akhil/spyne/projects/image_enhancement/temp/MZSR/GT/Set5"
save_path ="/home/akhil/spyne/projects/image_enhancement/temp/MZSR/Input/g20/Set5"
for file in os.listdir(base):
	img = cv2.imread(os.path.join(base,file))
	img = modcrop(img,scale)
	cv2.imwrite(os.path.join(save_path,file),img)
