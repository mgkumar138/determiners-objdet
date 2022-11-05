from backend.utils import saveload
import numpy as np
from backend.utils import compute_all_ious, create_output_txt, center_coord

determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
               "few", "both", "neither", "little", "much", "either", "our"]

[pred_bb,pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','test_predbb_out',1)
#create_output_txt(gdt=output_bb[:,:,:4], predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_gta/ori_gt')

gt_bb = np.copy(output_bb[:,:,:4])  # modified ground truth bounding box labels

# custom a evaluation
# - image: 3 apples, caption 'a apple', output: any 1 apple
# 1) get ground truth with all apples bounding box
# 2) compute IoU between predicted apple and all apples
# 3) select apple with highest IoU as ground truth and discard the rest. dont modify prediction
#
aidx = 0
alla = np.argmax(input_cap[:,:20],axis=1) == aidx
nidx = np.arange(len(pred_bb))[alla]
for n in nidx:
    objroi = np.argmax(input_cap[n,20:])
    objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

    # all correct answers
    allobjbb = input_bb[n,:,:4][objidx] * 256

    # Logic: use top 1 prediction for a
    top1idx = np.argsort(pred_score[n])[::-1][:1]
    predbb = pred_bb[n, top1idx]

    # check if predicted bb IoU with all 'apple' bb
    allious = compute_all_ious(allobjbb, predbb)
    gtbbsel = np.argmax(allious,axis=0)
    gtbb = allobjbb[gtbbsel]
    pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
    gt_bb[n] = pad_gtbb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_gta/mod_gt')


anyidx = 3
allany = np.argmax(input_cap[:,:20],axis=1) == anyidx
nidx = np.arange(len(pred_bb))[allany]
for n in nidx:
    objroi = np.argmax(input_cap[n,20:])
    objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

    # all coorect answers
    allobjbb = input_bb[n,:,:4][objidx] * 256

    # logic: use predictions with score above 0.5 for any
    critanyidx = np.arange(20)[pred_score[n]>0.5]
    predbb = pred_bb[n, critanyidx]

    # check if predicted bb IoU with all 'apple' bb
    allious = compute_all_ious(allobjbb, predbb)
    gtbbsel = np.argmax(allious,axis=0)
    gtbb = allobjbb[gtbbsel]
    pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
    gt_bb[n] = pad_gtbb


#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_any_a')


## this
thisidx = 7
allthis = np.argmax(input_cap[:,:20],axis=1) == thisidx
nidx = np.arange(len(pred_bb))[allthis]
for n in nidx:
    objroi = np.argmax(input_cap[n,20:])
    objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
    allobjbb = input_bb[n,:,:4][objidx] * 256

    # logic: object closer to camera with highest prediction score
    center = np.array([128.0,256.0])
    c = np.sqrt(256**2+256**2)
    this_thres = 154  # object must be bottom half of RGB image
    objdist = np.linalg.norm(center - center_coord(allobjbb) , axis=1)

    correct_this_bbs = allobjbb [objdist < this_thres]
    #target_bb = output_bb[n,:,:4]

    critthisidx = np.arange(20)[pred_score[n]>0.5]
    predbb = pred_bb[n, critthisidx]

    # check if predicted bb IoU with all 'apple' bb
    allious = compute_all_ious(correct_this_bbs, predbb)
    if len(allious) == 0:
        print(predbb)
        print(correct_this_bbs)
    gtbbsel = np.argmax(allious,axis=0)
    gtbb = allobjbb[gtbbsel]
    pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
    gt_bb[n] = pad_gtbb
#
# create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_this')

## that
thatidx = 8
allthat = np.argmax(input_cap[:,:20],axis=1) == thatidx
nidx = np.arange(len(pred_bb))[allthat]
for n in nidx:
    objroi = np.argmax(input_cap[n,20:])
    objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
    allobjbb = input_bb[n,:,:4][objidx] * 256

    # logic: object closer to camera with highest prediction score
    center = np.array([128.0,256.0])
    c = np.sqrt(256**2+256**2)
    this_thres = 128  # object must be bottom half of RGB image

    objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

    correct_that_bbs = allobjbb [objdist > this_thres]
    #target_bb = output_bb[n,:,:4]

    critthatidx = np.arange(20)[pred_score[n]>0.5]
    predbb = pred_bb[n, critthatidx]

    # check if predicted bb IoU with all 'apple' bb
    allious = compute_all_ious(correct_that_bbs, predbb)
    if len(allious) == 0:
        print(predbb)
        print(correct_that_bbs)
    gtbbsel = np.argmax(allious,axis=0)
    gtbb = allobjbb[gtbbsel]
    pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
    gt_bb[n] = pad_gtbb

create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_a_any_this_that')