import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("coco.names")
names = yolo.config._names

img_dir = "../dataset_dir/images/few/3250_few_orange_img.png"
img = Image.open(img_dir)
image_tensor = np.asarray(img)[None,:]


detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1")
detector_output = detector(image_tensor)
classes = detector_output["detection_classes"]
boxes = detector_output['detection_boxes']*256.0
scores = detector_output['detection_scores']

plt.figure(figsize=(4,4))
plt.imshow(image_tensor[0])
#plt.title(captions_array[ridx[i]])
for bb in range(boxes.shape[1]):
    if scores[0,bb] > 0.3:
        truerect = patches.Rectangle((boxes[0, bb, 1], boxes[0, bb, 0]), boxes[0, bb, 3]-boxes[0, bb, 1], boxes[0, bb, 2]-boxes[0, bb, 0],linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(truerect)
        plt.text(boxes[0, bb, 1], boxes[0, bb, 0], '{} {:.2g}'.format(names[int(classes[0,bb])], scores[0,bb]), color='g')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
plt.savefig('../Fig/fasterrcnn_analysis.png')