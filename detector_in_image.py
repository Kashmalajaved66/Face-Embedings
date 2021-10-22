import numpy as np
import cv2 as cv
import os


file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5  
## Detection with resnet 10
prototxt_file = file_path + 'Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('MobileNetSSD caffe model loaded successfully')
image = cv.imread(file_path + 'images' + os.sep + 'Frame_2.jpg')
origin_h, origin_w = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)

detections = net.forward()
print('Face detection accomplished')
## landmarks

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > threshold:
        bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
        
        x_start, y_start, x_end, y_end = bounding_box.astype('int')
        crop_img = image[y_start:y_end, x_start:x_end]
        origin_h1, origin_w1 = crop_img.shape[:2]
        print(origin_h1, origin_w1)
        blob1 = cv.dnn.blobFromImage(cv.resize(crop_img, (48 , 48)), 1.0, (48, 48), (104.0, 177.0, 123.0))
        prototxt_file1 = file_path + 'Landmarks/det3_relu.prototxt'
        caffemodel_file1 = file_path + 'Landmarks/det3_relu.caffemodel'
        net1 = cv.dnn.readNetFromCaffe(prototxt_file1, caffeModel=caffemodel_file1)
        net1.setInput(blob1)
        Landmarks = net1.forward()
        print(Landmarks)
        print(Landmarks[0][0]*origin_h1,Landmarks[0][5]*origin_w1)
        label = '{0:.2f}%'.format(confidence * 100)
        cv.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv.rectangle(image, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
        for i in range(5):
            cv.circle(crop_img, (int(Landmarks[0][i]*origin_h1*2),int(Landmarks[0][i+1]*origin_w1)), radius=1, color=(0, 0, 255), thickness=-1)
        cv.putText(image, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv.imshow('output', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()
