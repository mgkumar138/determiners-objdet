#!/usr/bin/env python
# coding: utf-8
# from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
# from PIL import Image
import tensorflow as tf
# import inflect
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten
from backend.utils import custom_loss, create_output_txt, saveload

# # Determiners Dataset baseline model

tfrecords_dir = "tfrecords"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

# ### Read tfrecords

# #### tfrecords dataset config

# In[24]:


ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

# #### read and parse tfrecords


determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
               "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each", "the"]


def parse_tfrecord_fn(example, labeled=True):
    feature_description = {
        "file_name": tf.io.FixedLenFeature([], tf.string),
        #         "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        "caption": tf.io.VarLenFeature(tf.string),
        "caption_one_hot": tf.io.VarLenFeature(tf.int64),
        "areas": tf.io.VarLenFeature(tf.int64),
        "category_ids": tf.io.VarLenFeature(tf.int64),
        "output_category_ids": tf.io.VarLenFeature(tf.int64),
        "output_areas": tf.io.VarLenFeature(tf.int64)
    }

    sequence_features = {
        "input_bboxes": tf.io.VarLenFeature(tf.int64),
        "output_bboxes": tf.io.VarLenFeature(tf.int64),
        "input_one_hot": tf.io.VarLenFeature(tf.int64),
        "output_one_hot": tf.io.VarLenFeature(tf.int64)
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=feature_description,
                                                            sequence_features=sequence_features)

    example = {**context, **sequence}
    for key in example.keys():
        if type(example[key]) == tf.sparse.SparseTensor:
            if (example[key].dtype == "string"):
                example[key] = tf.sparse.to_dense(example[key], default_value='b')
            else:
                example[key] = tf.sparse.to_dense(example[key])

    #     example["image"] = tf.io.decode_png(example["image"], channels=3)
    print(example["file_name"])
    raw = tf.io.read_file(example["file_name"])
    example["image"] = tf.io.decode_png(raw, channels=3)
    #     image = example["image"]
    #     print(example["caption_one_hot"])

    return example


def map_to_inputs(example):
    image = example["image"]
    caption = example["caption_one_hot"]
    input_bbox = example["input_bboxes"]
    input_label = example["category_ids"]
    output_labels = example["output_category_ids"]
    input_one_hot = tf.cast(example["input_one_hot"], dtype=tf.float64)
    output_bboxes = example["output_bboxes"]
    output_one_hot = tf.cast(example["output_one_hot"], dtype=tf.float64)
    caption_one_hot = example["caption_one_hot"]

    input_one_hot = tf.concat([input_one_hot[:, :4] / 256, input_one_hot[:, 4:]], axis=1)
    output_one_hot = tf.concat([output_one_hot[:, :4] / 256, output_one_hot[:, 4:]], axis=1)
    #     input_mask =
    output_mask = tf.stack(output_one_hot[:, 4])
    n_pad = 20 - tf.shape(output_labels)[0]
    output_labels_padded = tf.pad(output_labels, [[0, n_pad]], "CONSTANT")
    return (input_one_hot, caption_one_hot), (output_mask, caption_one_hot[:25])


# In[41]:


train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/train/*.tfrec")
print(train_filenames)

train_dataset = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
example = next(iter(train_dataset))

val_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/val/*.tfrec")
print(val_filenames)
val_dataset = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

# In[42]:


#print(example[0][1])

# In[43]:

### random bounding box selector model

maxbb = 20
nclass = 25
class SimpleBboxSelector(tf.keras.Model):
    def __init__(self):
        super(SimpleBboxSelector, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.ran_sel = tf.random.uniform

    def call(self,inputs):

        bbs = self.ran_sel(shape=(len(inputs[0]),maxbb), minval=0,maxval=1)
        det = tf.cast(inputs[1][:,:nclass],dtype=tf.float32)
        return [bbs, det]

# In[44]:

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = [tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()]
model = SimpleBboxSelector()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy','categorical_crossentropy'],run_eagerly=True)

# train model on dataset
epochs = 1
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
tr_loss = history.history['loss']
val_loss = history.history['val_loss']

print(model.summary())
#model.save_weights("train_ns_bb_cls_model_weights.h5")

# load & evaluate model on test datset

test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
print(test_filenames)
test_dataset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
test_dataset = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.evaluate(test_dataset)
preds = model.predict(test_dataset)
print(preds[1].shape)


test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
print(test_filenames)
test_dataset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
examples = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)

results = []

pred_bb = []
pred_cls = []
pred_score = []
gt_bb = []
gt_cls = []
all_bb = []
all_emb = []

for i, example in enumerate(examples):
    if i==155:
        print(example["input_one_hot"])
        anybb = example["input_one_hot"]
    inputs, outputs = map_to_inputs(example)

    inputs = tf.expand_dims(inputs[0], axis=0), tf.expand_dims(inputs[1], axis=0)
    pred = model(inputs)
    [pred_tr_score, pred_tr_cls] = pred[0].numpy(), pred[1].numpy()
    pred_ts_bb = (example["input_one_hot"].numpy()[:, :4] * (pred_tr_score > 0.5)[:, :, None])

    category_id = np.argmax(pred_tr_cls)
    #bboxes = example["input_one_hot"].numpy()[:, :4]

    pred_bb.append(pred_ts_bb)
    pred_cls.append(pred_tr_cls)
    pred_score.append(pred_tr_score)

    gt_bb.append(np.pad(example["output_bboxes"],((0,20-len(example["output_bboxes"])),(0,0))))
    gt_cls.append(inputs[1][0,:25])

    all_bb.append(np.pad(example["input_one_hot"],((0,20-len(example["input_one_hot"])),(0,0))))
    all_emb.append(example["caption_one_hot"])

    for i, mask in enumerate(list((pred_tr_score > 0.5)[0])):
        if mask:
            bbox = pred_ts_bb[0][i]
            results.append(
                {"image_id": int(example["image_id"].numpy()), "bbox": bbox.tolist(), "category_id": int(category_id),
                 "score": float(pred_tr_score[0][i])})

        #     break

print(results)

# transform ground truth to account multiple solutions

#modified_gt = transform_gt(original_gt)

# ann_dir = "annotations"
# split = "test"
# json.dump(modified_gt, open(os.path.join("ns_results", f"{split}_modified.json"), "w"))


results_dir = "rand_results"
split = "test"
json.dump(results, open(os.path.join("ns_results", f"{split}_results.json"), "w"))

# In[47]:


import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

dataDir = './'
dataType = 'test'
suffix = "annotations"
annFile = '%s/annotations/%s_%s.json' % (dataDir, dataType, suffix)
cocoGt = COCO(annFile)

resFile = '%s/ns_results/%s_results.json' % (dataDir, dataType)
cocoDt = cocoGt.loadRes(resFile)

annType = "bbox"

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

#create_output_txt(gdt=np.array(gt_bb), predt=np.vstack(pred_bb), confi=np.vstack(pred_score),gd_cls=np.array(gt_cls),pred_cls=np.vstack(pred_cls), directory='ns_cls/test_bb_cap')
#saveload('save','test_predbb_out_v2', [np.vstack(pred_bb),np.vstack(pred_cls),np.vstack(pred_score), np.array(all_bb), np.array(all_emb),np.array(gt_bb)])