import tensorflow as tf
import matplotlib.pyplot as plt
from backend.utils import saveload, train_val_test_split, create_output_txt, generate_img_caption_bb_mask
import numpy as np

images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_20000',1)

objdet_norm_bb = np.concatenate([objdet_bb[:,:,:4]/256.0, objdet_bb[:,:,4:]],axis=2)
target_norm_bb = np.concatenate([target_bb[:,:,:4]/256.0, target_bb[:,:,4:]],axis=2)

traindata, valdata, testdata = train_val_test_split(images, captions, target_norm_bb, objdet_norm_bb, sentemb_vec)
[tr_img, tr_cap, tr_out, tr_bb, tr_emb] = traindata
[val_img, val_cap, val_out, val_bb, val_emb] = valdata
[ts_img, ts_cap, ts_out, ts_bb, ts_emb] = testdata

tr_bb_rs, val_bb_rs,ts_bb_rs = np.reshape(tr_bb, (len(tr_bb),-1)), np.reshape(val_bb, (len(val_bb),-1)), np.reshape(ts_bb, (len(ts_bb),-1))

tr_out_rs = np.any(tr_out > 0,axis=2)*1
val_out_rs = np.any(val_out > 0,axis=2)*1
ts_out_rs = np.any(ts_out > 0,axis=2)*1

tr_cls_id = tr_emb[:,:20]
val_cls_id = val_emb[:,:20]
ts_cls_id = ts_emb[:,:20]

nemb = 64
nhid = 64
class GetTop(tf.keras.Model):
    def __init__(self):
        super(GetTop, self).__init__()
        self.cap_embed = tf.keras.layers.Dense(nemb,activation='relu', name='cap_emb')  # same shape as bb+lbl
        self.img_embed = tf.keras.layers.Dense(nemb, activation='relu', name='img_emb')  # same shape as bb+lbl
        self.dense1 = tf.keras.layers.Dense(nhid,activation='relu', name='fusion')
        #self.dense2 = tf.keras.layers.Dense(nhid, activation='relu')
        self.bb_mask = tf.keras.layers.Dense(tr_out_rs.shape[1],activation='sigmoid',name='bb_mask')
        self.det_class = tf.keras.layers.Dense(tr_cls_id.shape[1], activation='softmax', name='det_class')


    def call(self,inputs):
        bb = self.img_embed(tf.cast(inputs[0],dtype=tf.float32))
        c = self.cap_embed(tf.cast(inputs[1],dtype=tf.float32))
        bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
        #bb_c = tf.math.multiply(bb, c)  # concat/multiply/add
        #x = self.dense2(self.dense1(bb_c))
        x = self.dense1(bb_c)
        bbs = self.bb_mask(x)
        det = self.det_class(x)
        return [x,bbs, det]

evalmodel = GetTop()
evalmodel.build([(None,500),(None,40)])
#evalmodel.load_weights("train_ns_bb_cls_model_weights.h5")
evalmodel.summary()
[rfr,bbmask,detcl] = evalmodel.predict([tr_bb_rs,tr_emb])

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

scaler = StandardScaler()
scaler.fit(rfr)
X = scaler.transform(rfr)

lda = LinearDiscriminantAnalysis()
lda.fit(X, np.argmax(tr_cls_id,axis=1))
lda_trans = lda.transform(X)
coeff = lda.scalings_

pca = PCA()
pca_trans = pca.fit_transform(X)

tsne = TSNE()
tsn_trans = tsne.fit_transform(X)


alltrans = [lda_trans, pca_trans, tsn_trans]
labels = ['LD','PC','TSNE']
determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
               "few", "both", "neither", "little", "much", "either", "our"]
for p in range(3):
    plt.figure(figsize=(6, 6))
    xs = alltrans[p][:,0]
    ys = alltrans[p][:,1]
    plt.scatter(xs,ys, c = np.argmax(tr_cls_id,axis=1), cmap='gist_rainbow')

    for i in range(20):
        #idx = np.argmax(np.argmax(tr_cls_id, axis=1)==i)
        #plt.text(xs[idx], ys[idx], determiners[i], color='k', fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        centroid = np.mean(alltrans[p][np.argmax(tr_cls_id, axis=1)==i],axis=0)
        plt.text(centroid[0], centroid[1], determiners[i], color='k', fontsize=10, bbox=dict(facecolor='white', alpha=0.75))

    plt.xlabel('{}1'.format(labels[p]))
    plt.ylabel('{}2'.format(labels[p]))
    plt.title('Determiner representation')
    plt.savefig('../Fig/rand_det_rep_{}.png'.format(labels[p]))

plt.show()
