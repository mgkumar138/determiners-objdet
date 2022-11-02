import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

img_dir = "../dataset_dir/images/few/2882_img.png"
img = Image.open(img_dir)
image_tensor = np.asarray(img)[None,:]
#image_tensor = 2*(tf.image.resize(img, [512, 512])/255.0)-1

detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1")
detector_output = detector(image_tensor)
classes = detector_output["detection_classes"]
names = detector_output["detection_class_names"]
boxes = detector_output['detection_boxes']
scores = detector_output['detection_scores']

plt.figure(figsize=(4,4))
plt.imshow(img[0])
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
#plt.savefig('../Fig/efficientdet0_analysis.png')