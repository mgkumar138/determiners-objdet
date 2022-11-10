from backend.utils import saveload, train_val_test_split, create_output_txt, generate_img_caption_bb_mask
from backend.retinanet_backend import RetinaNetBoxLoss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#generate_img_caption_bb_mask(show_example=False)

images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_20000',1)

upsampany = False
if upsampany:
    anyidx = (np.argmax(sentemb_vec[:,:20],axis=1)==3)
    otheridx = anyidx^True
    idx = np.arange(len(anyidx))[anyidx]
    randidx = np.random.choice(np.arange(np.sum(anyidx)),np.sum(anyidx), replace=False)

    anyobj = objdet_bb[anyidx]
    anycap = sentemb_vec[anyidx]
    anygt = target_bb[anyidx]

    newobj = []
    newcap = []
    newgt = []
    for n in range(len(anyobj)):
        objs = []
        caps = []
        gts = []
        solns = np.argmax(anyobj[n,:,5:],axis=1) == np.argmax(anycap[n,20:])
        for s in np.arange(20)[solns]:
            objs.append(anyobj[n])
            caps.append(anycap[n])
            gts.append(np.pad(anyobj[n,s][None,:], ((0,19),(0,0))))

        newobj.append(np.array(objs))
        newcap.append(np.array(caps))
        newgt.append(np.array(gts))
    newobj = np.concatenate(newobj,axis=0)[randidx]
    newcap = np.concatenate(newcap,axis=0)[randidx]
    newgt = np.concatenate(newgt,axis=0)[randidx]


    objdet_bb = np.concatenate([objdet_bb[otheridx], newobj],axis=0)
    sentemb_vec = np.concatenate([sentemb_vec[otheridx], newcap],axis=0)
    target_bb = np.concatenate([target_bb[otheridx], newgt],axis=0)

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

# increase samples for any

# custom model
def custom_loss(y_true, y_pred):
    # classid
    # truecls = tf.argmax(y_true[:,20:],axis=1)
    # anyidx = (truecls == 3)  # any class
    # idx = (truecls != 3)
    #
    # bc_loss = tf.keras.metrics.binary_crossentropy(y_true[:,:20][idx], y_pred[:,:20][idx], from_logits=False)
    # cat_loss = tf.keras.metrics.binary_crossentropy(y_true[:,20:][idx], y_pred[:,20:][idx], from_logits=False)

    bc_loss = tf.keras.metrics.binary_crossentropy(y_true[:,:20], y_pred[:,:20], from_logits=False)
    cat_loss = tf.keras.metrics.binary_crossentropy(y_true[:,20:], y_pred[:,20:], from_logits=False)
    loss = tf.reduce_mean(bc_loss + cat_loss)
    return loss


nemb = 64
nhid = 64
trainwithcaptions = False
exptname = 'classbb_20det_1000exp_bb_clsblb_{}cap_1hid_concat_{}_{}N_bincross'.format(trainwithcaptions, nemb, nhid)

class SimpleDense(tf.keras.Model):
    def __init__(self):
        super(SimpleDense, self).__init__()
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
        return tf.concat([bbs, det],axis=1)

# train model
model = SimpleDense()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#loss_fn = [tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()] #tf.keras.losses.MeanSquaredError()
loss_fn = custom_loss #tf.keras.losses.MeanSquaredError()

epochs = 10
model.compile(optimizer=optimizer, loss=loss_fn, metrics='binary_crossentropy',run_eagerly=False)
if trainwithcaptions:
    history = model.fit(x=[tr_bb_rs,tr_emb],y=tf.concat([tr_out_rs, tr_cls_id],axis=1),
                        validation_data=([val_bb_rs,val_emb],tf.concat([val_out_rs, val_cls_id],axis=1)),
                        epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)
else:
    history = model.fit(x=[tr_bb_rs,np.zeros_like(tr_emb)],y=tf.concat([tr_out_rs, tr_cls_id],axis=1),
                        validation_data=([val_bb_rs,np.zeros_like(val_emb)],tf.concat([val_out_rs, val_cls_id],axis=1)),
                        epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)

print(model.summary())
#model.save_weights("train_ns_bb_cls_model_weights.h5")


# train data
tr_out[:,:,:4] *= 256.0
pred_tr_out = model.predict([tr_bb_rs,tr_emb])
[pred_tr_score, pred_tr_cls] = pred_tr_out[:,:20], pred_tr_out[:,20:]
pred_tr_bbcap = (tr_bb[:,:,:4] * (pred_tr_score> 0.5)[:,:,None]) * 256.0
train_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=tr_out_rs, y_pred=pred_tr_score, from_logits=False))
#create_output_txt(gdt=tr_out, predt=pred_tr_bbcap, confi=pred_tr_score,gd_cls=tr_cls_id,pred_cls=pred_tr_cls,directory='ns_cls_any/train_bb_cap')
#
# # train data
val_out[:,:,:4] *= 256.0
pred_val_out = model.predict([val_bb_rs,val_emb])
[pred_val_score, pred_val_cls] = pred_val_out[:,:20], pred_val_out[:,20:]
pred_val_bbcap = (val_bb[:,:,:4] * (pred_val_score> 0.5)[:,:,None]) * 256.0
val_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=val_out_rs, y_pred=pred_val_score, from_logits=False))
#create_output_txt(gdt=val_out, predt=pred_val_bbcap, confi=pred_val_score,gd_cls=val_cls_id,pred_cls=pred_val_cls, directory='ns_cls_any/val_bb_cap')

# test with both inputs
ts_out[:,:,:4] *= 256.0
pred_ts_out = model.predict([ts_bb_rs,ts_emb])
[pred_ts_score, pred_ts_cls] = pred_ts_out[:,:20], pred_ts_out[:,20:]
pred_ts_bbcap = (ts_bb[:,:,:4] * (pred_ts_score> 0.5)[:,:,None]) * 256.0
test_bcloss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=ts_out_rs, y_pred=pred_ts_score, from_logits=False))
create_output_txt(gdt=ts_out, predt=pred_ts_bbcap, confi=pred_ts_score,gd_cls=ts_cls_id,pred_cls=pred_ts_cls, directory='ns_cls_nocap/test_bb_cap')

# test with bb but no captions
pred_ts_out_npcap = model.predict([ts_bb_rs,np.zeros_like(ts_emb)])
[pred_ts_score_nocap, pred_ts_cls_nocap] = pred_ts_out_npcap[:,:20], pred_ts_out_npcap[:,20:]
pred_ts_bb = (ts_bb[:,:,:4] * (pred_ts_score_nocap> 0.5)[:,:,None]) * 256.0
test_bcloss_nocap = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=ts_out_rs, y_pred=pred_ts_score_nocap, from_logits=False))
create_output_txt(gdt=ts_out, predt=pred_ts_bb, confi=pred_ts_score_nocap,gd_cls=ts_cls_id,pred_cls=pred_ts_cls_nocap, directory='ns_cls_nocap/test_bb_only')

#print(testmap, testmapnocap)


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
