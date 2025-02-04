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

if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = input_dir
		#output_dir = "/home/antpc/Training_ALPR/alpr-unconstrained/test/"

		ocr_threshold = .4

		ocr_weights = bytes('data/ocr/ocr-net.weights', encoding="utf-8")
		ocr_netcfg  = bytes('data/ocr/ocr-net.cfg',  encoding="utf-8")
		ocr_dataset = bytes('data/ocr/ocr-net.data',  encoding="utf-8")

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
		#imgs_paths = sorted([os.path.abspath(os.path.join('/home/antpc/Training_ALPR/alpr-unconstrained/try', p)) for p in os.listdir('/home/antpc/Training_ALPR/alpr-unconstrained/try')])

		print('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, bytes(img_path,encoding='utf-8'),thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				#print("Before sorting: {}".format(L))

				L.sort(key=lambda x: x.tl()[0])
				#L.sort(key=lambda x: x.tl()[1])


				#print(lambda x: x.tl()[0],L)

				lp_str = ''.join([chr(l.cl()) for l in L])
				print(*L, sep='\n')

				#with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					#f.write(lp_str + '\n')

				print('\t\tLP: %s' % lp_str)

			else:

				print('No characters found')

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
