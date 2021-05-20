import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms
import os
from Levenshtein import distance

if __name__ == '__main__':

	try:
	
		#input_dir  = sys.argv[1]
		#output_dir = input_dir
		output_dir = "/home/antpc/Training_ALPR/alpr-unconstrained/test/"

		ocr_threshold = .4

		ocr_weights = bytes('data/ocr/ocr-net.weights', encoding="utf-8")
		ocr_netcfg  = bytes('data/ocr/ocr-net.cfg',  encoding="utf-8")
		ocr_dataset = bytes('data/ocr/ocr-net.data',  encoding="utf-8")

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		#imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
		imgs_paths = sorted([os.path.abspath(os.path.join('data_cropped/All_cropped', p)) 
									for p in os.listdir('data_cropped/All_cropped')])
		
		label_file = open(r"data_cropped/all_label/labels.txt", "r")
		data = label_file.readlines()
		comp_file = open(r"data_cropped/all_label/comparision.txt", "w")
		comp_file.write("Absolute" +"  :  " + "Predicted" + "\n")
		t_per = 0
		p_acc = 0

		print('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):

			# print('\tScanning %s' % img_path)

			diff_char = 0

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, bytes(img_path,encoding='utf-8'),thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				#print("Before sorting: {}".format(L))

				# L.sort(key=lambda x: x.tl()[0])
				#L.sort(key=lambda x: x.tl()[1])
				points = list(map(lambda x: [chr(x.cl()), [int(round(float(x.tl()[0])*1000)),int(round(float(x.tl()[1])*1000))],[int(round(float(x.br()[0])*1000)),int(round(float(x.br()[1])*1000))]], L))
				x_y_cordinate_char = points
				#listofNum = list(filter(lambda x : x > 10 and x < 20, listofNum))

				line1 = list(filter(lambda x: abs(x_y_cordinate_char[0][1][1] - x[1][1]) <= 100,points))
				line2 = list(filter(lambda x: abs(x_y_cordinate_char[0][1][1] - x[1][1]) > 100,points))

				line1.sort(key=lambda x: x[1][0])
				line2.sort(key=lambda x: x[1][0])

				#print(lambda x: x.tl()[0],L)

				# lp_str = ''.join([chr(l.cl()) for l in L])
				# print(*L, sep='\n')

				# print('Points Customized are: {}\n'.format(points))

				# print("\nSeperate lines are:\n")
				# print(line1)
				# print(line2)

				if len(line1)>0 and len(line2)>0:
					if line1[0][1][1] > line2[0][1][1]:
						line1, line2 = line2, line1

				final_list = line1 + line2
				# print(final_list)

				lp_str = ''.join(list(map(lambda x: x[0], final_list)))
				len_st = len(lp_str)

				if(len_st >= 4 and lp_str[3] == 'I'):
					lp_str = lp_str[0:3]+"1"+lp_str[4:]
				if(len_st >= 5 and lp_str[4] == 'I'):
					lp_str = lp_str[0:4]+"1"+lp_str[5:]
				if(len_st >= 6 and lp_str[5] == 'I'):
					lp_str = lp_str[0:5]+"1"+lp_str[6:]
				if(len_st >= 7 and lp_str[6] == 'I'):
					lp_str = lp_str[0:6]+"1"

				#with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					#f.write(lp_str + '\n')

				o_plt = data[i].split("\n")[0]

				diff_char = distance(o_plt, lp_str)
				comp_file.write(o_plt+"  :  "+ lp_str+"\n")
				t_per = t_per + (diff_char/len(o_plt))*100
				if(diff_char == 0):
					p_acc += 1

				# print('\t\tLP: %s\n\n' % lp_str)

			else:

				print('No characters found')
		final_loss = t_per/len(imgs_paths)
		print("Avg Character level Accuracy Score on all type Plates are: {} %".format(100-final_loss))
		print("Avg plate level Accuracy Score on all type Plates are: {} %".format((p_acc/len(imgs_paths))*100))
		comp_file.close()
		label_file.close()

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
