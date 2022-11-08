from backend.utils import saveload, get_USE_model, train_val_test_split, create_output_txt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#matchbb = match_bb()

images, captions, target_bb, objdet_bb, _, sentemb_vec = saveload('load','../dataset_dir/img_cap_bb_mask_matrix_20000',1)

objdet_norm_bb = np.concatenate([objdet_bb[:,:,:4]/256.0, objdet_bb[:,:,4:]],axis=2)
target_norm_bb = np.concatenate([target_bb[:,:,:4]/256.0, target_bb[:,:,4:]],axis=2)

traindata, valdata, testdata = train_val_test_split(images, captions, target_norm_bb, objdet_norm_bb, sentemb_vec)
[tr_img, tr_cap, tr_out, tr_bb, tr_emb] = traindata
[val_img, val_cap, val_out, val_bb, val_emb] = valdata
[ts_img, ts_cap, ts_out, ts_bb, ts_emb] = testdata

def random_proposal_selector(objdet_bb, true_bb, proposal_threshold=0.5):
    N = len(objdet_bb)
    all_bb = objdet_bb[:,:,:4]
    all_proposals = objdet_bb[:, :, 4]
    det_prob = np.random.uniform(low=0, high=1, size=(N, objdet_bb.shape[1]))
    #sort_prob = -np.sort(-det_prob,axis=1)
    det_score = det_prob *(det_prob > proposal_threshold)
    #proposal_selection = all_proposals * (proposal_prob > proposal_threshold)
    bb_selection = all_bb * (det_prob > proposal_threshold)[:,:,None] * 256.0

    #loss = box_loss(y_pred=bb_selection, y_true=true_bb[:,:,:4])
    return bb_selection, det_score

# train dataset
train_pred_bb, train_score = random_proposal_selector(tr_bb, tr_out)
val_pred_bb, val_score = random_proposal_selector(val_bb, val_out)
test_pred_bb, test_score = random_proposal_selector(ts_bb, ts_out)

tr_out[:,:,:4] *= 256.0
val_out[:,:,:4] *= 256.0
ts_out[:,:,:4] *= 256.0
create_output_txt(gdt=tr_out, predt=train_pred_bb, confi=train_score, directory='rand/train_bb_only')
create_output_txt(gdt=val_out, predt=val_pred_bb, confi=val_score, directory='rand/val_bb_only')
create_output_txt(gdt=ts_out, predt=test_pred_bb, confi=test_score, directory='rand/test_bb_only')

