import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect
from src.label import dknet_label_conversion
from src.utils import nms
import os
from Levenshtein import distance


if __name__ == '__main__':

    try:

        #input_dir  = sys.argv[1]
        #output_dir = input_dir
        output_dir = "/home/antpc/Training_ALPR/alpr-unconstrained/test/"

        ocr_threshold = .4

        ocr_weights = bytes('data/ocr/ocr-net.weights', encoding="utf-8")
        ocr_netcfg = bytes('data/ocr/ocr-net.cfg',  encoding="utf-8")
        ocr_dataset = bytes('data/ocr/ocr-net.data',  encoding="utf-8")

        ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
        ocr_meta = dn.load_meta(ocr_dataset)

        #imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
        imgs_paths = sorted([os.path.abspath(os.path.join('data_cropped/only_one_lined', p))
                             for p in os.listdir('data_cropped/only_one_lined')])
        label_file = open(r"data_cropped/labels.txt", "r")
        data = label_file.readlines()
        comp_file = open(r"data_cropped/comp.txt", "w")
        comp_file.write("Absolute" +"  :  " + "Predicted" + "\n")
        print('Performing OCR...')
        t_per = 0
        p_acc = 0

        for i, img_path in enumerate(imgs_paths):

            # print('\tScanning %s' % img_path)
            diff_char = 0

            bname = basename(splitext(img_path)[0])

            R, (width, height) = detect(ocr_net, ocr_meta, bytes(
                img_path, encoding='utf-8'), thresh=ocr_threshold, nms=None)

            if len(R):

                L = dknet_label_conversion(R, width, height)
                L = nms(L, .45)
                #print("Before sorting: {}".format(L))

                L.sort(key=lambda x: x.tl()[0])
                #L.sort(key=lambda x: x.tl()[1])

                #print(lambda x: x.tl()[0],L)

                lp_str = ''.join([chr(l.cl()) for l in L])
                len_st = len(lp_str)
                
                # print(*L, sep='\n')
                # with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                #f.write(lp_str + '\n')
               
                if(len_st >= 4 and lp_str[3] == 'I'):
                    lp_str = lp_str[0:3]+"1"+lp_str[4:]
                if(len_st >= 5 and lp_str[4] == 'I'):
                    lp_str = lp_str[0:4]+"1"+lp_str[5:]
                if(len_st >= 6 and lp_str[5] == 'I'):
                    lp_str = lp_str[0:5]+"1"+lp_str[6:]
                if(len_st >= 7 and lp_str[6] == 'I'):
                    lp_str = lp_str[0:6]+"1"

                o_plt = data[i].split(" ")[1].split("\n")[0]

                diff_char = distance(o_plt, lp_str)
                comp_file.write(o_plt+"  :  "+ lp_str+"\n")
                t_per = t_per + (diff_char/len(o_plt))*100
                if(diff_char == 0):
                    p_acc += 1

                #print('\t\tLP: %s' % lp_str)

            else:

                print('No characters found')
        final_loss = t_per/len(imgs_paths)
        print("Avg Character level Accuracy Score on Single Lined Plates are: {} %".format(100-final_loss))
        print("Avg plate level Accuracy Score on Single Lined Plates are: {} %".format((p_acc/len(imgs_paths))*100))
        comp_file.close()
        label_file.close()

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
