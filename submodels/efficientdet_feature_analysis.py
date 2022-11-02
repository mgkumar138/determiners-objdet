import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
# from yolov4.tf import YOLOv4
#
# yolo = YOLOv4()
#
# yolo.config.parse_names("coco.names")
# names = yolo.config._names

img_dir = "../dataset_dir/images/few/3250_few_orange_img.png"
img = Image.open(img_dir)
img = np.asarray(img)[None,:]
image_tensor = 2*(tf.image.resize(img, [320, 320])/255.0)-1


class EfficientDetNetTrainHub(tf.keras.Model):
    """EfficientDetNetTrain for Hub module."""
    def __init__(self):
        super(EfficientDetNetTrainHub, self).__init__()
        self.base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
        # class/box output prediction network.
        num_anchors = 100
        conv2d_layer = tf.keras.layers.Conv2D
        self.classes = conv2d_layer(
            80 * num_anchors,
            kernel_size=3,
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            padding='same',
            name='class_net/class-predict')
        self.boxes = tf.keras.layers.Conv2D(
            filters=4 * num_anchors,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_size=3,
            activation=None,
            bias_initializer=tf.zeros_initializer(),
            padding='same',
            name='box_net/box-predict')


    def call(self, inputs, training):
        cls_outputs, box_outputs = self.base_model(inputs, training=False)
        for i in range(self.config.max_level - self.config.min_level + 1):
            cls_outputs[i] = self.classes(cls_outputs[i])
            box_outputs[i] = self.boxes(box_outputs[i])
        return cls_outputs, box_outputs

model = EfficientDetNetTrainHub()
cls, bb = model(image_tensor, training=False)



# plt.figure(figsize=(8,8))
# for t,th in enumerate([0.6,0.4,0.2,0]):
#     plt.subplot(2,2,t+1)
#     plt.imshow(img[0])
#     #plt.title(captions_array[ridx[i]])
#     for bb in range(box_outputs.shape[1]):
#         if cls_outputs[0,bb] > th:
#             truerect = patches.Rectangle((boxes[0, bb, 1], boxes[0, bb, 0]), boxes[0, bb, 3]-boxes[0, bb, 1], boxes[0, bb, 2]-boxes[0, bb, 0],linewidth=2, edgecolor='g', facecolor='none')
#             plt.gca().add_patch(truerect)
#             plt.text(boxes[0, bb, 1], boxes[0, bb, 0], '{} {:.2g}'.format(names[int(classes[0,bb])], scores[0,bb]), color='g')
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# plt.show()
# plt.savefig('../Fig/efficientdet0_analysis.png')