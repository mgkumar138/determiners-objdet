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



# label_encoder = LabelEncoder()
#
# label_encoder.encode_batch(batch_images=tf.cast(tr_img[100:120],dtype=tf.float32),
#                            gt_boxes=tf.cast(tr_out[100:120,:,:4]*256.0,dtype=tf.float32),
#                            cls_ids=tr_out[100:120,:,5:])







#
# perf = 'MSE Loss Train = {:.2g}, Val = {:.2g}, Test = {:.2g}'.format(train_bc, val_bc, test_bc)
# print(perf)
# exptname = 'random_bb_selector'
#
# ts_out[:,:,:4] *= 256.0
# test_pred_bb[:,:,:4] *= 256.0
#
# f = plt.figure(figsize=(8,6))
# f.text(0.01,0.01,exptname, fontsize=8)
# f.suptitle('Random bb selector ' + perf)
#
# for i in range(12):
#     plt.subplot(3,4,i+1)
#     plt.imshow(ts_img[i])
#     plt.title(ts_cap[i], fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
#
#     for bb in range(ts_out.shape[1]):
#         if ts_out[i,bb,4] > 0.5:
#             truerect = patches.Rectangle((ts_out[i, bb, 0], ts_out[i, bb, 1]), ts_out[i, bb, 2], ts_out[i, bb, 3],linewidth=2, edgecolor='g', facecolor='none')
#             plt.gca().add_patch(truerect)
#
#         if test_pred_bb[i,bb,4] > 0.5:
#             predrect = patches.Rectangle((test_pred_bb[i,bb,0], test_pred_bb[i,bb,1]), test_pred_bb[i,bb,2], test_pred_bb[i,bb,3], linewidth=2, edgecolor='r', facecolor='none')
#             plt.gca().add_patch(predrect)
#
# plt.tight_layout()
# plt.show()
# #f.savefig('../Fig/'+exptname+'.png')