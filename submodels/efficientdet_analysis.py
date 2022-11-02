import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("coco.names")
names = yolo.config._names

#detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/1")
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")

# image = ...  # A batch of preprocessed images with shape [batch_size, height, width, 3].
# base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
# cls_outputs, box_outputs = base_model(image, training=training)

img_dir = "../dataset_dir/images/few/3250_few_orange_img.png"
img = Image.open(img_dir)
img = np.asarray(img)[None,:]
#image_tensor = 2*(tf.image.resize(img, [512, 512])/255.0)-1

boxes, scores, classes, num_detections = detector(img)  # [ymin, xmin, ymax, xmax]
#classes, boxes = detector(tf.cast(image_tensor,dtype=tf.float32))

plt.figure(figsize=(8,8))
for t,th in enumerate([0.6,0.4,0.2,0]):
    plt.subplot(2,2,t+1)
    plt.imshow(img[0])
    #plt.title(captions_array[ridx[i]])
    for bb in range(boxes.shape[1]):
        if scores[0,bb] > th:
            truerect = patches.Rectangle((boxes[0, bb, 1], boxes[0, bb, 0]), boxes[0, bb, 3]-boxes[0, bb, 1], boxes[0, bb, 2]-boxes[0, bb, 0],linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(truerect)
            plt.text(boxes[0, bb, 1], boxes[0, bb, 0], '{} {:.2g}'.format(names[int(classes[0,bb])], scores[0,bb]), color='g')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
plt.savefig('../Fig/efficientdet0_analysis.png')