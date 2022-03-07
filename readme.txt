Update Points
1. Added the ability to handle videos that exist in all folders under the whole folder and generate the corresponding folder paths.

2. Fixed the previous problem that deepSORT performed abnormal matching.

3. Use the feature map extracted from the 20th layer of YOLO to replace the pre-trained Re-ID feature extractor of deepSORT.

Before you run the tracker
Make sure that you fulfill all the requirements: 
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7. To install, run:

pip install -r requirements.txt


How to run it 
## For folder case 
## The output will be saved in the current folder of inference/2021 as the following example

python track.py --source /home/natsu/AIST/2021 --yolo_weights /home/natsu/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5x.pt --save-txt --classes 0 2 --show-vid --save-vid --img-size 1280 --conf-thres 0.1

## For single video case 
## The output will be saved in the current folder of inference/outputs as the following example
python track.py --source /home/natsu/AIST/video/test.mp4 --yolo_weights /home/natsu/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5x.pt --save-txt --classes 0 2 --show-vid --save-vid --img-size 1280 --conf-thres 0.1

# --source
#  video path 
# /home/natsu/AIST/video/2021-07-06-17-30-01_Tsukuba.mp4  

# --yolo_weights 
# yolov5 model path  default yolov5s.pt

# --img-size 640 or 1280 
# The size of the feature map of the image can be specified when the YOLO model is processed (must be a multiple of 32)

# --conf-thres 
# Threshold for the presence of an object in each anchor

# --save-txt 
# Store the location of the tracked object in each frame

# --save-vid
# save video result after yolo+deepsort

# --classes 
# yolo specifies the category of the detected object (total 80 classes)
# 0 2 is person and car (other can check with coco dataset) 　

# --show-vid
# You can dynamically check the results