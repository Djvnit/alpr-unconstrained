import os
from PIL import Image
from os import walk
import linecache 
ipath = sorted([os.path.abspath(os.path.join('/home/antpc/Training_ALPR/images', p)) for p in os.listdir('/home/antpc/Training_ALPR/images')])
lpath = sorted([os.path.abspath(os.path.join('/home/antpc/Training_ALPR/labels', p)) for p in os.listdir('/home/antpc/Training_ALPR/labels')])
_, _, filenames = next(walk('/home/antpc/Training_ALPR/images'))
filenames = sorted(filenames)
crop_p = "/home/antpc/Training_ALPR/data_cropped/only_one_lined/"
file = open("/home/antpc/Training_ALPR/data_cropped/only_one_lined/labels.txt","w")
for i in range(0,len(ipath)):
    rcline = linecache.getline(lpath[i],8) 
    reqc = list(rcline.split(" "))
    reqc[4] = list(reqc[4].split('\n'))[0]
    req_crop = [int(reqc[1]),int(reqc[2]),int(reqc[1])+int(reqc[3]),int(reqc[2])+int(reqc[4])]
    img = Image.open(ipath[i])
    img2 = img.crop(req_crop)
    img2.save(crop_p+filenames[i])
     
    rst_line = linecache.getline(lpath[i], 7) 
    req_st = list(rst_line.split(" "))
    req_st = "".join(list(req_st[1].split("-")))

    req_st = req_st.rstrip("\n")
    file.write(filenames[i]+', '+ req_st + "\n")
    #file.write(req_st+ "\n")
file.close()
