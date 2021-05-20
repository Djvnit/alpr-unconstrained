# ALPR in Unscontrained Scenarios

## Introduction

This repository contains the author's implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

* Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

## Requirements

* In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. 
* The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:
* Go to alpr_unconstrained folder throgh your terminal and then use command

```shellscript
cd darknet && make
```

**The current version was tested in an Ubuntu 20.04 machine, with Keras 2.3.4, TensorFlow 2.4.1, OpenCV 4.5.1, NumPy 1.19.5 and Python 3.8.5**

## Download Models

After building the Darknet framework, you must execute the "get-networks.sh" script. This will download all the trained models:

```shellscript
bash get-networks.sh
```

## Running a simple test

Use the script "run.sh" to run our ALPR approach. It requires 3 arguments:
* __Input directory (-i):__ should contain at least 1 image in JPG or PNG format;
* __Output directory (-o):__ during the recognition process, many temporary files will be generated inside this directory and erased in the end. The remaining files will be related to the automatic annotated image;
* __CSV file (-c):__ specify an output CSV file.

```shellscript
$ bash get-networks.sh && bash run.sh -i samples/test -o outputs -c outputs/results.csv
```
*Note:* _Don't remove -i, -o, -c from the command just trace the path used and run the command

## Above were the instructions for testing both Detector + OCR

# For only testing OCR of the above model  
### Run plates_ocr.py
#### In plates_ocr.py 
  > The path of image folder is stored in imgs_paths.

  > Lables.txt is opened for reading the actual labels by of each image that is obtained by command
  ```shellscript
   python data_processing.py
   ```
  In *data_processing.py* we have used the training set of *[UFPR-ALPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/)* dataset for testing the robustness the OCR engine. 
  * Just trace the path of input folders in specified files to run the OCR without distruptions.   
  * UFPR-ALPR dataset(total 4500 images) is divided into 3 parts i.e. training (40% of images), tesing (40% of images), validation (20% of images.  
  * Training set also contains the actual characters on license plate because of which we have used it for testing the accuracy of OCR engine.  
  * comparision.txt is created for writing both the actual and predicted characters on the number plate. Which will help you to visualize the actual and predicted string at a same place side by side.

  ## For testing the accuracy the accuracy of OCR on single lined plate (1440 images) use command
  ```shellscript 
  python license-plate-ocr_one_lined_plates.py
  ```
  ## Output
  ```
   > The average character level accuracy on 1440 one lined license plate images are about 96.4%.

  > The plate level accuracy on 1400 one lined license plate images are about 78%.
  ```
  ## For testing the accuracy the accuracy of OCR on mixed (both single and double lined) license plate (1800 images) use command
  ```shellscript 
  python plates_ocr.py
  ```
  ## Output
  ```
  > The average character level accuracy on 1800 general license plate images are about 90.000%.
  
  > The plate level accuracy on 1800 general license plate images are about 64.0%.
  ```

## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.
