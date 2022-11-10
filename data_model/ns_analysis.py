#!/usr/bin/env python
# coding: utf-8
# from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
import numpy as np
import os
# from PIL import Image
import tensorflow as tf
# import inflect
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten
import pickle
#from backend.utils import custom_loss, create_output_txt, saveload

def saveload(opt, name, variblelist):
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var

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
    caption = example["caption"]
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


train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
print(train_filenames)

train_dataset = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
example = next(iter(train_dataset))

# val_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/val/*.tfrec")
# print(val_filenames)
# val_dataset = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
# val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
# val_dataset = val_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
# val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
# val_dataset = val_dataset.batch(BATCH_SIZE)
#
# test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
# print(test_filenames)
# test_dataset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
# test_dataset = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
# test_dataset = test_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# In[42]:


#print(example[0][1])

# In[43]:

### neural bounding box selector model architecture

nemb = 128
nhid = 256
maxbb = 20
nclass = 0
class SimpleBboxSelector(tf.keras.Model):
    def __init__(self):
        super(SimpleBboxSelector, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.img_embed = tf.keras.layers.Dense(nemb, activation='relu', name='img_emb')  # same shape as bb+lbl
        self.cap_embed = tf.keras.layers.Dense(nemb,activation='relu', name='cap_emb')  # same shape as bb+lbl

        self.dense1 = tf.keras.layers.Dense(nhid,activation='relu', name='combine')
        self.dense2 = tf.keras.layers.Dense(nhid, activation='relu')
        self.bb_mask = tf.keras.layers.Dense(maxbb,activation='sigmoid',name='bb_mask')
        #self.det_class = tf.keras.layers.Dense(nclass, activation='softmax', name='det_class')

    def call(self,inputs):
        bb = self.img_embed(tf.cast(self.flatten(inputs[0]),dtype=tf.float32))
        c = self.cap_embed(tf.cast(inputs[1],dtype=tf.float32))
        bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
        # bb_c = tf.math.multiply(bb, c)  # concat/multiply/add
        x = self.dense2(self.dense1(bb_c))
        #x = self.dense1(bb_c)
        bbs = self.bb_mask(x)
        #det = self.det_class(x)
        return [x, bbs]

# In[44]:
model = SimpleBboxSelector()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = [tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy','categorical_crossentropy'],run_eagerly=True)
model.build([(None,420),(None,41)])

[prerfr,_] = model.predict(train_dataset)

if nclass == 25:
    model.load_weights("model_bb_det_weights.h5")
    dtype = 'det'

elif nclass == 16:
    model.load_weights("model_bb_obj_weights.h5")
    dtype = 'obj'
else:
    model.load_weights("ns_mw_det_noun.h5")
    dtype = 'only'
#model.summary()

[postrfr,_] = model.predict(train_dataset)


examples = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
examples = examples.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)

det_cls = []
for example in examples:
    inputs, outputs = map_to_inputs(example)
    det_cls.append(inputs[1][:25])
det_cls = np.array(det_cls)

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# lda = LinearDiscriminantAnalysis()
# lda.fit(X, np.argmax(det_cls,axis=1))
# lda_trans = lda.transform(X)
# coeff = lda.scalings_
#
# pca = PCA()
# pca_trans = pca.fit_transform(X)




allrfr = [prerfr, postrfr]
determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
               "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
               "the"]

alltrans = []
for p in range(2):
    rfr = allrfr[p]
    scaler = StandardScaler()
    scaler.fit(rfr)
    X = scaler.transform(rfr)

    tsne = TSNE(perplexity=8,learning_rate='auto',init='pca')
    tsn_trans = tsne.fit_transform(X)
    alltrans.append(tsn_trans)


titles = ['before','after']
f,ax = plt.subplots(ncols=1, nrows=3,figsize=(4, 8))
allcentroids = np.zeros([2,25,2])
for p in range(2):
    xs = alltrans[p][:,0]
    ys = alltrans[p][:,1]

    ax[p].scatter(xs, ys, c=np.argmax(det_cls,axis=1), cmap='hsv')

    # ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # for i in range(25):
    #     idx = np.argmax(det_cls,axis=1)==i
    #     plt.scatter(xs[idx],ys[idx], label=determiners[i])

    for i in range(25):
        #idx = np.argmax(np.argmax(tr_cls_id, axis=1)==i)
        #plt.text(xs[idx], ys[idx], determiners[i], color='k', fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        allcentroids[p,i] = np.mean(alltrans[p][np.argmax(det_cls, axis=1)==i],axis=0)[:2]
        ax[p].annotate(determiners[i], xy=allcentroids[p,i],color='k', fontsize=8, bbox=dict(facecolor='white', alpha=0.3))
    #plt.legend(determiners)

    ax[p].set_xlabel('tSNE_1')
    ax[p].set_ylabel('tSNE_2')
    ax[p].set_title('Embedding {} training'.format(titles[p]), fontsize=10)
    ax[p].spines.right.set_visible(False)
    ax[p].spines.top.set_visible(False)
    #adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)



import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as scp
SimMatrix = np.matmul(allcentroids[1], allcentroids[1].T)/(np.linalg.norm(allcentroids[1],)**2)
Zw = scp.linkage(1-SimMatrix)
ax[2].set_title('Clustering determiner centroids', fontsize=10)
ax[2] = scp.dendrogram(Zw, labels=determiners,leaf_font_size=8, leaf_rotation=90)

f.tight_layout()

#plt.savefig('../Fig/tsne_bbonly_vert.jpg'.format(dtype))
plt.show()
