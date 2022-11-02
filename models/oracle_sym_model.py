from backend.utils import saveload, get_USE_model, train_val_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_5750',1)

objdet_norm_bb = np.concatenate([objdet_bb[:,:,:4]/256.0, objdet_bb[:,:,4:]],axis=2)
target_norm_bb = np.concatenate([target_bb[:,:,:4]/256.0, target_bb[:,:,4:]],axis=2)

traindata, valdata, testdata = train_val_test_split(images, captions, target_norm_bb, objdet_norm_bb, sentemb_vec)
[tr_img, tr_cap, tr_out, tr_bb, tr_emb] = traindata
[val_img, val_cap, val_out, val_bb, val_emb] = valdata
[ts_img, ts_cap, ts_out, ts_bb, ts_emb] = testdata

tr_out_rs = np.any(tr_out > 0,axis=2)*1
val_out_rs = np.any(val_out > 0,axis=2)*1
ts_out_rs = np.any(ts_out > 0,axis=2)*1

ts_out[:,:,:4] *= 256.0

def oracle_proposal_selector(objdet_bb, true_bb_class, proposal_threshold=0.5):
    N = len(objdet_bb)
    all_bb = objdet_bb[:,:,:4]
    all_proposals = objdet_bb[:, :, 4]
    proposal_prob = true_bb_class

    proposal_selection = all_proposals * (proposal_prob > proposal_threshold)
    bb_selection = all_bb * (proposal_prob > proposal_threshold)[:,:,None]
    bincx_bb = tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(y_true=true_bb_class, y_pred=proposal_selection, from_logits=False))

    return np.concatenate([bb_selection*256.0,proposal_selection[:,:,None]],axis=2), bincx_bb

# train dataset
train_pred_bb, train_bc = oracle_proposal_selector(tr_bb, tr_out_rs)
val_pred_bb, val_bc = oracle_proposal_selector(val_bb, val_out_rs)
test_pred_bb, test_bc = oracle_proposal_selector(ts_bb, ts_out_rs)

perf = 'Log Loss Train = {:.2g}, Val = {:.2g}, Test = {:.2g}'.format(train_bc, val_bc, test_bc)
print(perf)
exptname = 'classbb_oracle_proposal_selector'

f = plt.figure(figsize=(8,6))
f.text(0.01,0.01,exptname, fontsize=8)
f.suptitle('Random bb selector ' + perf)

for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(ts_img[i])
    plt.title(ts_cap[i], fontsize=8)
    plt.xticks([])
    plt.yticks([])

    for bb in range(ts_out.shape[1]):
        if ts_out[i,bb,4] > 0.5:
            truerect = patches.Rectangle((ts_out[i, bb, 0], ts_out[i, bb, 1]), ts_out[i, bb, 2], ts_out[i, bb, 3],linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(truerect)

        if test_pred_bb[i,bb,4] > 0.5:
            predrect = patches.Rectangle((test_pred_bb[i,bb,0], test_pred_bb[i,bb,1]), test_pred_bb[i,bb,2], test_pred_bb[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(predrect)

plt.tight_layout()
plt.show()
#f.savefig('../Fig/'+exptname+'.png')