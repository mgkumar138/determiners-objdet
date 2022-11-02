import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os
from backend.utils import saveload, convert_to_corners, build_head, LabelEncoder, RetinaNetLoss, preprocess_data


# dataset
# images, _, target_bb, _, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_5000',1)
#
# #target_bb: N x 20 x 4bb + 1 prob + 20class
# images = tf.cast(images, dtype=tf.float32)
# sentemb_vec = tf.cast(sentemb_vec, dtype=tf.float32)
# #target_bb = tf.concat([target_bb[:,:,:4],target_bb[:,:,5:]],axis=-1)
# target_bb = tf.cast(target_bb, dtype=tf.float32)

(train_dataset, val_dataset), dataset_info = tfds.load("coco/2017", split=["train", "validation"], with_info=True, data_dir="data")

batch_size = 1
label_encoder = LabelEncoder()
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

# model

class RetinaNet(tf.keras.Model):
    def __init__(self, num_classes=20):
        super(RetinaNet, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(4,3,1,'same')
        #self.c2 = tf.keras.layers.Conv2D(4, 3, 1, 'same')
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, inputs):
        feature1 = self.c1(inputs)
        #feature2 = self.c2(inputs)
        features = [feature1]
        N = tf.shape(inputs[0])[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


num_classes = 20
epochs = 1
#label_encoder = LabelEncoder()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes)
model_dir = "simpleobjdet/"
callbacks_list = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss", save_best_only=False, save_weights_only=True, verbose=2,)]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=loss_fn, optimizer=optimizer,run_eagerly=True)

model.fit(train_dataset.take(1), batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
          verbose=2)


# inference
#predictions = model([images, sentemb_vec], training=False)