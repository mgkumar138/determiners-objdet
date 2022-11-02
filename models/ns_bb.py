from backend.utils import saveload, train_val_test_split, create_output_txt, generate_img_caption_bb_mask
from backend.retinanet_backend import RetinaNetBoxLoss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#generate_img_caption_bb_mask(show_example=False)

images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_20000',1)

objdet_norm_bb = np.concatenate([objdet_bb[:,:,:4]/256.0, objdet_bb[:,:,4:]],axis=2)
target_norm_bb = np.concatenate([target_bb[:,:,:4]/256.0, target_bb[:,:,4:]],axis=2)

traindata, valdata, testdata = train_val_test_split(images, captions, target_norm_bb, objdet_norm_bb, sentemb_vec)
[tr_img, tr_cap, tr_out, tr_bb, tr_emb] = traindata
[val_img, val_cap, val_out, val_bb, val_emb] = valdata
[ts_img, ts_cap, ts_out, ts_bb, ts_emb] = testdata

# reshape bb and target
tr_bb_rs, val_bb_rs,ts_bb_rs = np.reshape(tr_bb, (len(tr_bb),-1)), np.reshape(val_bb, (len(val_bb),-1)), np.reshape(ts_bb, (len(ts_bb),-1))
#tr_bb_rs, val_bb_rs,ts_bb_rs = np.reshape(tr_bb[:,:,5:], (len(tr_bb),-1)), np.reshape(val_bb[:,:,5:], (len(val_bb),-1)), np.reshape(ts_bb[:,:,5:], (len(ts_bb),-1))
#tr_out_rs, val_out_rs,ts_out_rs = np.reshape(tr_out, (len(tr_out),-1)), np.reshape(val_out, (len(val_out),-1)), np.reshape(ts_out, (len(ts_out),-1))

tr_out_rs = np.any(tr_out > 0,axis=2)*1
val_out_rs = np.any(val_out > 0,axis=2)*1
ts_out_rs = np.any(ts_out > 0,axis=2)*1

tr_cls_id = tr_emb[:,:20]
val_cls_id = val_emb[:,:20]
ts_cls_id = ts_emb[:,:20]

# img input: 4 coordinates + 1 probability + 20 class label
# caption input: 20 determinres + 20 nouns
# output: 20 possible outputs

# custom model
smoothl1loss = RetinaNetBoxLoss()
nemb = 64
nhid = 64
trainwithcaptions = True
exptname = 'classbb_20det_1000exp_bb_clsblb_{}cap_2hid_mul_{}_{}N_bincross'.format(trainwithcaptions, nemb, nhid)

class SimpleDense(tf.keras.Model):
    def __init__(self):
        super(SimpleDense, self).__init__()
        self.cap_embed = tf.keras.layers.Dense(nemb,activation='relu', name='cap_emb')  # same shape as bb+lbl
        self.img_embed = tf.keras.layers.Dense(nemb, activation='relu', name='img_emb')  # same shape as bb+lbl
        self.dense1 = tf.keras.layers.Dense(nhid,activation='relu', name='fusion')
        #self.dense2 = tf.keras.layers.Dense(nhid, activation='relu')
        self.out_bb_mask = tf.keras.layers.Dense(tr_out_rs.shape[1],activation='sigmoid',name='bb_class')

    def call(self,inputs):
        bb = self.img_embed(tf.cast(inputs[0],dtype=tf.float32))
        c = self.cap_embed(tf.cast(inputs[1],dtype=tf.float32))
        bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
        #bb_c = tf.math.multiply(bb, c)  # concat/multiply/add
        #x = self.dense2(self.dense1(bb_c))
        x = self.dense1(bb_c)
        bbs = self.out_bb_mask(x)
        return bbs

# train model
model = SimpleDense()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy() #tf.keras.losses.MeanSquaredError()

epochs = 20
model.compile(optimizer=optimizer, loss=loss_fn, metrics='binary_crossentropy',run_eagerly=False)
if trainwithcaptions:
    history = model.fit(x=[tr_bb_rs,tr_emb],y=tr_out_rs,validation_data=([val_bb_rs,val_emb],val_out_rs), epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)
else:
    history = model.fit(x=[tr_bb_rs,np.zeros_like(tr_emb)],y=tr_out_rs,validation_data=([val_bb_rs,np.zeros_like(val_emb)],val_out_rs), epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)

print(model.summary())
model.save_weights("train_nsmodel_weights.h5")


# train data
tr_out[:,:,:4] *= 256.0
pred_tr_score = model.predict([tr_bb_rs,tr_emb])
pred_tr_bbcap = (tr_bb[:,:,:4] * (pred_tr_score> 0.5)[:,:,None]) * 256.0
#tr_loss = smoothl1loss.call(y_true=tr_out[:,:,:4],y_pred=pred_tr_bbcap)
train_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=tr_out_rs, y_pred=pred_tr_score, from_logits=False))
create_output_txt(gdt=tr_out, predt=pred_tr_bbcap, confi=pred_tr_score, directory='ns/train_bb_cap')

# train data
val_out[:,:,:4] *= 256.0
pred_val_score = model.predict([val_bb_rs,val_emb])
pred_val_bbcap = (val_bb[:,:,:4] * (pred_val_score> 0.5)[:,:,None]) * 256.0
val_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=val_out_rs, y_pred=pred_val_score, from_logits=False))
create_output_txt(gdt=val_out, predt=pred_val_bbcap, confi=pred_val_score, directory='ns/val_bb_cap')

# test with both inputs
ts_out[:,:,:4] *= 256.0
pred_ts_score = model.predict([ts_bb_rs,ts_emb])
pred_ts_bbcap = (ts_bb[:,:,:4] * (pred_ts_score> 0.5)[:,:,None]) * 256.0
test_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=ts_out_rs, y_pred=pred_ts_score, from_logits=False))
create_output_txt(gdt=ts_out, predt=pred_ts_bbcap, confi=pred_ts_score, directory='ns/test_bb_cap')

# test with bb but no captions
pred_ts_score_nocap = model.predict([ts_bb_rs,np.zeros_like(ts_emb)])
pred_ts_bb = (ts_bb[:,:,:4] * (pred_ts_score_nocap> 0.5)[:,:,None]) * 256.0
test_bcloss_nocap = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=ts_out_rs, y_pred=pred_ts_score_nocap, from_logits=False))
create_output_txt(gdt=ts_out, predt=pred_ts_bb, confi=pred_ts_score_nocap, directory='ns/test_bb_only')

#print(testmap, testmapnocap)

#
#
#
# toplayer = tf.keras.Sequential(toplayer)
# toplayer.predict([ts_bb_rs,ts_emb])
# testmodel.summary()

# class GetTop(tf.keras.Model):
#     def __init__(self):
#         super(GetTop, self).__init__()
#         testmodel = SimpleDense()
#         testmodel.build([(None, 500), (None, 40)])
#         testmodel.load_weights("train_nsmodel_weights.h5")
#         self.layer = testmodel.layers[:-1]
#
#     def call(self,inputs):
#         bb = self.img_embed(tf.cast(inputs[0],dtype=tf.float32))
#         c = self.cap_embed(tf.cast(inputs[1],dtype=tf.float32))
#         #bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
#         bb_c = tf.math.multiply(bb, c)  # concat/multiply/add
#         #x = self.dense2(self.dense1(bb_c))
#         x = self.dense1(bb_c)
#         bbs = self.out_bb_mask(x)
#         return bbs


# resize bb to image pixel


#ious = all_iou(predbb, truebb)

# plot prediction
f,ax = plt.subplots(2,7,figsize=(12,4))
f.text(0.01,0.01,exptname, fontsize=8)
ax[0,0].plot(history.history['loss'])
ax[0,0].plot(history.history['val_loss'])
ax[0,0].legend(['Train','Val'])
ax[0,0].set_title('CapBC={:.2g}'.format(test_bcloss), fontsize=12)
ax[1,0].set_title('NoCapBC={:.2g}'.format(test_bcloss_nocap), fontsize=12)
ax[1,0].axis('off')
ax[0,0].set_ylabel('Binary Crossentropy loss')
ax[0,0].set_xlabel('Epoch')

for j in range(2):
    for i in range(6):
        #plt.subplot(2,4,i+2)
        #plt.axis([0,256,0,256])
        ax[j,i+1].imshow(ts_img[i])
        ax[j,i+1].set_title(ts_cap[i], fontsize=8)
        ax[j, i + 1].set_xticks([])
        ax[j, i + 1].set_yticks([])

        for bb in range(ts_out.shape[1]):
            if ts_out[i,bb,4] > 0.5:
                truerect = patches.Rectangle((ts_out[i, bb, 0], ts_out[i, bb, 1]), ts_out[i, bb, 2], ts_out[i, bb, 3],linewidth=2, edgecolor='g', facecolor='none')
                ax[j,i+1].add_patch(truerect)

            if j == 1:
                if pred_ts_score_nocap[i,bb] > 0.5:
                    predrect = patches.Rectangle((pred_ts_bb[i,bb,0], pred_ts_bb[i,bb,1]), pred_ts_bb[i,bb,2], pred_ts_bb[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
                    ax[j,i+1].add_patch(predrect)
            else:
                if pred_ts_score[i,bb] > 0.5:
                    predrect = patches.Rectangle((pred_ts_bbcap[i,bb,0], pred_ts_bbcap[i,bb,1]), pred_ts_bbcap[i,bb,2], pred_ts_bbcap[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
                    ax[j,i+1].add_patch(predrect)

plt.tight_layout()
plt.show()
#f.savefig('../Fig/'+exptname+'.png')