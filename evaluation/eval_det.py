from backend.utils import saveload
import numpy as np
from backend.utils import compute_all_ious, create_output_txt, center_coord


# Should be everything except, all, every, much, little, your, our, my

# custom a evaluation
# - image: 3 apples, caption 'a apple', output: any 1 apple
# 1) get ground truth with all apples bounding box
# 2) compute IoU between predicted apple and all apples
# 3) select apple with highest IoU as ground truth and discard the rest. dont modify prediction

def main_change_gt_multiple_soln(gt_bb, input_bb, input_cap, pred_score, pred_bb):
    gt_bb = a_an_either_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0, 1, 18,24])
    gt_bb = any_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3)
    gt_bb = this_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7)
    gt_bb = that_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=8)
    gt_bb = these_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9)
    gt_bb = those_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10)
    gt_bb = some_many_few_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[11,12,13],  lowerbound=[5,8,2])
    return gt_bb




def some_many_few_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb,detidx=[11,12,13], lowerbound=[5,8,2]):
    for det, lb in zip(detidx, lowerbound):
        # some 5-6, many 8-9, few 2-3
        alldet = np.argmax(input_cap[:,:25],axis=1) == det # captions with some
        allcountobj = np.argmax(input_cap[alldet, 25:], axis=1) < 10  # captions with countables
        # index of captions with some and countables
        nidx = np.arange(len(pred_bb))[alldet]
        nidx = nidx[allcountobj]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

            # all targets
            allobjbb = input_bb[n,:,:4][objidx] #* 256

            # logic: use predictions with score above 0.5 for any
            critanyidx = np.arange(20)[pred_score[n]>0.5]
            predbb = pred_bb[n, critanyidx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbb = allobjbb[gtbbsel]
            if not (lb-1)<len(gtbb)<(lb+2):
                remainsoln = np.delete(np.arange(len(allobjbb)), gtbbsel)
                idx = np.random.choice(remainsoln, lb-len(predbb), replace=False)
                gtbb = np.concatenate([gtbb, allobjbb[idx]],axis=0) # atleast 5 bounding boxes

            assert (lb-1)<len(gtbb)<(lb+2), print('some, many, few wrong gt')
            pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb


def a_an_either_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0,1,18]):
    # a, an, either
    for aidx in detidx:
        alla = np.argmax(input_cap[:,:25],axis=1) == aidx
        nidx = np.arange(len(pred_bb))[alla]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,:,:4][objidx] #* 256

            # Logic: use top 1 prediction for a
            top1idx = np.argsort(pred_score[n])[::-1][:1]
            predbb = pred_bb[n, top1idx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbb = allobjbb[gtbbsel]
            #assert len(gtbb) == 1
            if len(gtbb) !=1:
                print(gtbb)
            pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_gta/mod_gt')


def both_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=14):
    # both
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

        # all coorect answers
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # Logic: use top 2 prediction for a
        top2idx = np.argsort(pred_score[n])[::-1][:2]
        predbb = pred_bb[n, top2idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)
        gtbbsel = np.argmax(allious,axis=0)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 2
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb

def any_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3):
    # any
    allany = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allany]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

        # all coorect answers
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # logic: use predictions with score above 0.5 for any
        critanyidx = np.arange(20)[pred_score[n]>0.5]
        predbb = pred_bb[n, critanyidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)
        gtbbsel = np.argmax(allious,axis=0)
        gtbb = allobjbb[gtbbsel]
        assert 0 < len(gtbb) <= len(allobjbb)
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def this_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7):
    ## this
    allthis = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthis]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
        c = np.sqrt(256**2+256**2)
        this_thres = 154  # object must be bottom half of RGB image
        objdist = np.linalg.norm(center - center_coord(allobjbb) , axis=1)

        correct_this_bbs = allobjbb [objdist < this_thres]
        #target_bb = output_bb[n,:,:4]

        top1idx = np.argsort(pred_score[n])[::-1][:1]
        predbb = pred_bb[n, top1idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_this_bbs, predbb)
        if len(allious) < 1:
            print(predbb)
            print(correct_this_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def that_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## that
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
        c = np.sqrt(256**2+256**2)
        this_thres = 128  # object must be bottom half of RGB image

        objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_that_bbs = allobjbb [objdist > this_thres]
        #target_bb = output_bb[n,:,:4]

        top1idx = np.argsort(pred_score[n])[::-1][:1]
        predbb = pred_bb[n, top1idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_that_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_that_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def these_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## these
    allthese = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthese]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
        this_thres = 154  # object must be bottom half of RGB image

        objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_that_bbs = allobjbb [objdist < this_thres]
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

        # if len(correct_that_bbs) != len(predbb):
        #     print(np.argmax(input_bb[n, :, 5:], axis=1))
        #     print(len(correct_that_bbs), len(predbb))
        #     print('next')

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_that_bbs)), gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_that_bbs[idx]],axis=0)
            print('adding soln to these')

        assert len(gtbb) >1
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def those_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10):
    ## those
    allthose = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthose]
    for n in nidx:
        objroi = np.argmax(input_cap[n,25:])
        objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
        allobjbb = input_bb[n,:,:4][objidx] #* 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
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

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_that_bbs)), gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_that_bbs[idx]],axis=0)
            print('adding soln to those')
        assert len(gtbb) > 1
        pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_a_an_either_any_both_this_that_these_those_2')


if __name__ == '__main__':
    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]

    [pred_bb,pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_out_v2',1)
    gt_bb = np.copy(output_bb[:,:,:4])  # modified ground truth bounding box labels

    modgt_bb = main_change_gt_multiple_soln(gt_bb, input_bb, input_cap, pred_score, pred_bb)

    #create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:25],pred_cls=pred_cls, directory='../data_model/ns_cls_modgt/mod_gt')

# your
# anyidx = 6
# allany = np.argmax(input_cap[:,:20],axis=1) == anyidx
# nidx = np.arange(len(pred_bb))[allany]
# for n in nidx:
#     objroi = np.argmax(input_cap[n,20:])
#     objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
#
#     # all targets
#     allobjbb = input_bb[n,:,:4][objidx] * 256
#     # target of interest: base of item on tray opposite
#     ythes = 128
#     targets = allobjbb[allobjbb[:,1]<ythes]
#
#     #target_bb = output_bb[n,:,:4]
#
#     gtbb = targets
#     pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
#     gt_bb[n] = pad_gtbb


# all santiy check that code is correct
# anyidx = 2
# allany = np.argmax(input_cap[:,:20],axis=1) == anyidx
# nidx = np.arange(len(pred_bb))[allany]
# for n in nidx:
#     objroi = np.argmax(input_cap[n,20:])
#     objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
#
#     # all coorect answers
#     allobjbb = input_bb[n,:,:4][objidx] * 256
#
#     # logic: use predictions with score above 0.5 for any
#     critanyidx = np.arange(20)[pred_score[n]>0.5]
#     predbb = pred_bb[n, critanyidx]
#
#     if len(allobjbb) != len(predbb):
#         print(np.argmax(input_bb[n, :, 5:], axis=1))
#         print(len(allobjbb), len(predbb))
#         print('next')
#
#     # check if predicted bb IoU with all 'apple' bb
#     allious = compute_all_ious(allobjbb, predbb)
#     gtbbsel = np.arange(len(allobjbb)) #np.argmax(allious,axis=1)
#     gtbb = allobjbb[gtbbsel]
#     pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
#     gt_bb[n] = pad_gtbb

