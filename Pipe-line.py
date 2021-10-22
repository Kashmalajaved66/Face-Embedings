import numpy as np
import argparse
import cv2
import time
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
threshold = 0.5 
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# load the input image from disk
image = cv2.imread(args["image"])
origin_h, origin_w = image.shape[:2]
print(origin_h, origin_w)
# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
print(classes)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (480 , 270)), 1.0,
	(480 , 270), (104.0, 177.0, 123.0))
print("input" , blob.shape)
net.setInput(blob)
start = time.time()
detections = net.forward()
print(detections.shape)
end = time.time()
print("[INFO] detection took {:.5} seconds".format(end - start))
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2 ]
    
    if confidence > threshold:
        print(confidence)
        bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
        x_start, y_start, x_end, y_end = bounding_box.astype('int')

        label = '{0:.2f}%'.format(confidence * 100)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        # 画文字的填充矿底色
        cv2.rectangle(image, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
        cv2.putText(image, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
