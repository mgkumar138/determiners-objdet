from backend.utils import saveload, get_USE_model, all_iou, generate_img_caption_bb_mask, train_val_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(0)

generate_img_caption_bb_mask(show_example=False)

images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_5750',1)

objdet_norm_bb = np.concatenate([objdet_bb[:,:,:4]/256.0, objdet_bb[:,:,4:]],axis=2)
target_norm_bb = np.concatenate([target_bb[:,:,:4]/256.0, target_bb[:,:,4:]],axis=2)

traindata, valdata, testdata = train_val_test_split(images, captions, target_norm_bb, objdet_norm_bb, sentemb_vec)
[tr_img, tr_cap, tr_out, tr_bb, tr_emb] = traindata
[val_img, val_cap, val_out, val_bb, val_emb] = valdata
[ts_img, ts_cap, ts_out, ts_bb, ts_emb] = testdata

# reshape bb and target
tr_bb_rs, val_bb_rs,ts_bb_rs = np.reshape(tr_bb, (len(tr_bb),-1)), np.reshape(val_bb, (len(val_bb),-1)), np.reshape(ts_bb, (len(ts_bb),-1))
tr_out_rs, val_out_rs,ts_out_rs = np.reshape(tr_out, (len(tr_out),-1)), np.reshape(val_out, (len(val_out),-1)), np.reshape(ts_out, (len(ts_out),-1))


# custom model
nemb = 256
nhid = 512
trainwithcaptions = False
exptname = '20det_250exp_bb_{}cap_2hid_relu_concat_{}_{}N'.format(trainwithcaptions, nemb, nhid)

class SimpleDense(tf.keras.Model):
    def __init__(self):
        super(SimpleDense, self).__init__()
        self.cap_embed = tf.keras.layers.Dense(nemb,activation='relu')  # same shape as bb+lbl
        self.img_embed = tf.keras.layers.Dense(nemb, activation='relu')  # same shape as bb+lbl
        self.dense1 = tf.keras.layers.Dense(nhid,activation='relu')
        self.dense2 = tf.keras.layers.Dense(nhid, activation='relu')
        self.out_bb_mask = tf.keras.layers.Dense(int(tr_out.shape[1]*tr_out.shape[2]),activation='sigmoid',name='bb_mask')

    def call(self,inputs):
        bb = self.img_embed(tf.cast(inputs[0],dtype=tf.float32))
        c = self.cap_embed(tf.cast(inputs[1],dtype=tf.float32))
        bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
        #bb_c = tf.math.add(bb, c)  # concat/multiply/add
        x = self.dense2(self.dense1(bb_c))
        #x = self.dense1(bb_c)
        bbs = self.out_bb_mask(x)
        return bbs

# train model
model = SimpleDense()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
epochs = 2
model.compile(optimizer=optimizer, loss=loss_fn, metrics='mse',run_eagerly=False)
if trainwithcaptions:
    history = model.fit(x=[tr_bb_rs,tr_emb],y=tr_out_rs,validation_data=([val_bb_rs,val_emb],val_out_rs), epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)
else:
    history = model.fit(x=[tr_bb_rs,np.zeros_like(tr_emb)],y=tr_out_rs,validation_data=([val_bb_rs,np.zeros_like(val_emb)],val_out_rs), epochs=epochs, batch_size=64, validation_split=0.0, shuffle=True)

#history = model.fit(x=[train_obj_bb,train_cap],y=train_target_bb, epochs=epochs, batch_size=64, validation_split=0.2, shuffle=True)
print(model.summary())

# test with both inputs
pred_bb_cap = model.predict([ts_bb_rs,ts_emb])
test_mse_bb_cap = np.mean((pred_bb_cap-ts_out_rs)**2)

# test with bb but no captions
pred_bb = model.predict([ts_bb_rs,np.zeros_like(ts_emb)])
test_mse_bb = np.mean((pred_bb-ts_out_rs)**2)

# resize bb to image pixel
ts_out[:,:,:4] *= 256.0

predbbcap = np.reshape(pred_bb_cap, newshape=ts_out.shape)
predbbcap[:,:,:4] *= 256.0

predbb = np.reshape(pred_bb, newshape=ts_out.shape)
predbb[:,:,:4] *= 256.0

#ious = all_iou(predbb, truebb)

# plot prediction
f,ax = plt.subplots(2,7,figsize=(12,4))
f.text(0.01,0.01,exptname, fontsize=8)
ax[0,0].plot(history.history['loss'])
ax[0,0].plot(history.history['val_loss'])
ax[0,0].legend(['Train','Val'])
ax[0,0].set_title('CapTestMSE={:.2g}'.format(test_mse_bb_cap), fontsize=10)
ax[1,0].set_title('NoCapTestMSE={:.2g}'.format(test_mse_bb), fontsize=10)
ax[1,0].axis('off')
ax[0,0].set_ylabel('MSE')
ax[0,0].set_xlabel('Epoch')

for j in range(2):
    for i in range(6):
        #plt.subplot(2,4,i+2)
        #plt.axis([0,256,0,256])
        ax[j,i+1].imshow(ts_img[i])
        ax[j,i+1].set_title(ts_cap[i])
        ax[j, i + 1].set_xticks([])
        ax[j, i + 1].set_yticks([])

        for bb in range(ts_out.shape[1]):
            if ts_out[i,bb,4] > 0.75:
                truerect = patches.Rectangle((ts_out[i, bb, 0], ts_out[i, bb, 1]), ts_out[i, bb, 2], ts_out[i, bb, 3],linewidth=2, edgecolor='g', facecolor='none')
                ax[j,i+1].add_patch(truerect)

            if j == 1:
                if predbb[i,bb,4] > 0.75:
                    predrect = patches.Rectangle((predbb[i,bb,0], predbb[i,bb,1]), predbb[i,bb,2], predbb[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
                    ax[j,i+1].add_patch(predrect)
            else:
                if predbbcap[i,bb,4] > 0.75:
                    predrect = patches.Rectangle((predbbcap[i,bb,0], predbbcap[i,bb,1]), predbbcap[i,bb,2], predbbcap[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
                    ax[j,i+1].add_patch(predrect)

plt.tight_layout()
plt.show()
#f.savefig('../Fig/'+exptname+'.png')