#from backend.utils import saveload
import numpy as np
#from backend.utils import compute_all_ious, create_output_txt, center_coord
import itertools
import os
import operator
import json
from collections import defaultdict


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if np.isnan(iou):
        print(iou)
    # return the intersection over union value
    return iou

def xywh_to_coord(boxes):
    return np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], axis=-1)


def compute_all_ious(gtbb,predbb):
    allious = np.zeros([len(gtbb), len(predbb)])
    for i in range(len(gtbb)):
        for j in range(len(predbb)):
            allious[i,j] = bb_intersection_over_union(xywh_to_coord(gtbb[i]), xywh_to_coord(predbb[j]))
    return allious

def center_coord(boxes):
    return np.concatenate([boxes[..., :2] + boxes[..., 2:]/2], axis=-1)


# Should be everything except, all, every, much, little, your, our, my

# custom a evaluation
# - image: 3 apples, caption 'a apple', output: any 1 apple
# 1) get ground truth with all apples bounding box
# 2) compute IoU between predicted apple and all apples
# 3) select apple with highest IoU as ground truth and discard the rest. dont modify prediction

import itertools
import operator


# def result_json_to_matrix(filename, max_bboxes = 20):
#     results = json.load(open(os.path.join("results", filename)))
#     get_attr = operator.attrgetter('image_id')
# #     print(results)
#     img_id_grouped = [list(g) for k, g in itertools.groupby(results, lambda x: x["image_id"])]
#     bboxes = list(map(lambda arr: list(map(lambda y: np.array(y["bbox"]), arr)),img_id_grouped))
#     prediction_s

#     for i, group in enumerate(bboxes):
#         for j in range(len(bboxes), group)
#             bboxes

def generate_corrected_gt_json_v2(gt_dir, results_dir, max_bboxes=20):
    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]
    results = json.load(open(os.path.join(results_dir)))
    get_attr = operator.attrgetter('image_id')
    #     print(results)
    img_id_grouped = [list(g) for k, g in itertools.groupby(results, lambda x: x["image_id"])]
    bboxes = list(map(lambda arr: list(map(lambda y: np.array(y["bbox"]), arr)), img_id_grouped))
    scores = list(map(lambda arr: list(map(lambda y: y["score"], arr)), img_id_grouped))
    labels = list(map(lambda arr: list(map(lambda y: y["category_id"], arr)), img_id_grouped))

    print(len(bboxes))
    print(len(scores))
    print(len(labels))
    for i, group in enumerate(bboxes):
        for j in range(len(group), max_bboxes):
            bboxes[i].append(np.array([0, 0, 0, 0]))
            scores[i].append(0)
            labels[i].append(0)
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])
        labels[i] = np.array(scores[i])
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    labels = np.array(scores)

    inputs = json.load(open(os.path.join(gt_dir)))
    input_annotations = inputs["input_oracle_annotations"]
    categories = inputs["categories"]
    gt_annotations = inputs["annotations"]
    image_idx = defaultdict(int)
    idx_to_id = defaultdict(int)
    gt_bboxes = [[] for i in inputs["images"]]
    input_bboxes = [[] for i in inputs["images"]]
    inputs['images'] = sorted(inputs['images'], key=lambda x: x['id'])
    for i, image in enumerate(inputs["images"]):
        image_idx[image['id']] = i
        idx_to_id[i] = [image['id']]

    for ann in gt_annotations:
        gt_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"]))
        #print(image_idx[ann["image_id"]])
    for ann in input_annotations:
        category_one_hot = [0 for i in range(len(categories))]
        category_one_hot[ann["category_id"]] = 1
        input_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"] + [1] + category_one_hot))

    for i, group in enumerate(gt_bboxes):
        for j in range(len(group), max_bboxes):
            gt_bboxes[i].append(np.array([0, 0, 0, 0]))

    for i, group in enumerate(input_bboxes):
        for j in range(len(group), max_bboxes):
            input_bboxes[i].append(np.array([0, 0, 0, 0] + [0] + [0 for i in range(len(categories))]))

    input_bboxes = np.array([np.array(row) for row in input_bboxes])
    gt_bboxes = np.array([np.array(row) for row in gt_bboxes])

    categories = inputs["categories"]
    category_one_hot = [0 for i in range(len(categories))]
    captions = []

    for i, img in enumerate(inputs["images"]):
        caption = img["caption"]
        det = caption.split()[0]
        noun = " ".join(caption.split()[1:])
        determiners.index(det)
        noun_id = 0
        if noun[-1] == "s":
            noun = noun[:-1]
        for cat in categories:
            if cat["name"] == noun:
                noun_id = cat["id"]
        det_one_hot = [0 for i in range(len(determiners))]
        det_one_hot[determiners.index(det)] = 1
        noun_one_hot = [0 for i in range(len(categories))]
        noun_one_hot[noun_id] = 1
        caption_one_hot = det_one_hot + noun_one_hot

        captions.append(caption_one_hot)

    captions = np.array(captions)
    print("ground truth bboxes shape: ", gt_bboxes.shape)
    print("input bboxes shape: ", input_bboxes.shape)
    print("scores shape: ", scores.shape)
    print("captions shape", captions.shape)

    gt_bboxes = main_change_gt_multiple_soln(gt_bboxes, input_bboxes, captions, scores, bboxes, max_bboxes)

    print(len(gt_bboxes))
    print(len(gt_annotations))
    new_bboxes = []
    new_annotations = []
    count = 0
    for i, img_bboxes in enumerate(gt_bboxes):
        #print(inputs["images"][i]["id"])
        for bbox in img_bboxes:
            bbox = list(bbox)
            bbox = list(map(np.int, bbox))
            if bbox == [0, 0, 0, 0]:
                continue
            else:
                new_bboxes.append(bbox)
                new_annotations.append({
                    "id": count,
                    "image_id": inputs["images"][i]["id"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "category_id": 1,
                    "iscrowd": 0
                })
            count += 1
    #print(new_annotations)
    print(len(new_bboxes))
    print("num of new_annotations", len(new_annotations))
    new_json = {
        "licenses": inputs["licenses"],
        "categories": inputs["categories"],
        "images": inputs["images"],
        "annotations": new_annotations
    }

    json.dump(new_json, open(f"./annotations/mod_test_annotations.json", "w"))


def generate_corrected_gt_json_v1(gt_dir, results_dir, max_bboxes=100):
    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]
    results = json.load(open(os.path.join(results_dir)))
    get_attr = operator.attrgetter('image_id')
    #     print(results)
    img_id_grouped = [list(g) for k, g in itertools.groupby(results, lambda x: x["image_id"])]
    bboxes = list(map(lambda arr: list(map(lambda y: np.array(y["bbox"]), arr)), img_id_grouped))
    scores = list(map(lambda arr: list(map(lambda y: y["score"], arr)), img_id_grouped))
    labels = list(map(lambda arr: list(map(lambda y: y["category_id"], arr)), img_id_grouped))

    print(len(bboxes))
    print(len(scores))
    print(len(labels))
    for i, group in enumerate(bboxes):
        for j in range(len(group), max_bboxes):
            bboxes[i].append(np.array([0, 0, 0, 0]))
            scores[i].append(0)
            labels[i].append(0)
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])
        labels[i] = np.array(scores[i])
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    labels = np.array(scores)

    inputs = json.load(open(os.path.join(gt_dir)))
    input_annotations = inputs["input_oracle_annotations"]
    categories = inputs["categories"]
    gt_annotations = inputs["annotations"]
    image_idx = defaultdict(int)
    gt_bboxes = [[] for i in inputs["images"]]
    input_bboxes = [[] for i in inputs["images"]]
    for i, image in enumerate(inputs["images"]):
        image_idx[image['id']] = i

    for ann in gt_annotations:
        gt_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"]))
    for ann in input_annotations:
        category_one_hot = [0 for i in range(len(categories))]
        category_one_hot[ann["category_id"]] = 1
        input_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"] + [1] + category_one_hot))

    for i, group in enumerate(gt_bboxes):
        for j in range(len(group), max_bboxes):
            gt_bboxes[i].append(np.array([0, 0, 0, 0]))

    for i, group in enumerate(input_bboxes):
        for j in range(len(group), max_bboxes):
            input_bboxes[i].append(np.array([0, 0, 0, 0] + [0] + [0 for i in range(len(categories))]))

    input_bboxes = np.array([np.array(row) for row in input_bboxes])
    gt_bboxes = np.array([np.array(row) for row in gt_bboxes])

    categories = inputs["categories"]
    category_one_hot = [0 for i in range(len(categories))]
    captions = []

    for i, img in enumerate(inputs["images"]):
        caption = img["caption"]
        det = caption.split()[0]
        noun = " ".join(caption.split()[1:])
        determiners.index(det)
        noun_id = 0
        if noun[-1] == "s":
            noun = noun[:-1]
        for cat in categories:
            if cat["name"] == noun:
                noun_id = cat["id"]
        det_one_hot = [0 for i in range(len(determiners))]
        det_one_hot[determiners.index(det)] = 1
        noun_one_hot = [0 for i in range(len(categories))]
        noun_one_hot[noun_id] = 1
        caption_one_hot = det_one_hot + noun_one_hot

        captions.append(caption_one_hot)

    captions = np.array(captions)
    print("ground truth bboxes shape: ", gt_bboxes.shape)
    print("input bboxes shape: ", input_bboxes.shape)
    print("scores shape: ", scores.shape)
    print("captions shape", captions.shape)

    gt_bboxes = main_change_gt_multiple_soln(gt_bboxes, input_bboxes, captions, scores, bboxes, max_bboxes)
    print(len(gt_bboxes))
    print(len(gt_annotations))

    return gt_bboxes


def main_change_gt_multiple_soln(gt_bb, input_bb, input_cap, pred_score, pred_bb, max_bb=20):
    #gt_bb = np.copy(gt_bb[:,:, :4])
    #gt_bb = my_your_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[5,6])
    gt_bb = a_an_either_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0, 1, 18,24])
    gt_bb = any_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3)
    gt_bb = this_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7)
    gt_bb = that_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=8)
    gt_bb = these_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9)
    gt_bb = those_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10)
    gt_bb = some_many_few_several_half_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[11,12,13, 21,22],  lowerbound=[5,8,2,4,-1], upperbound=[6,9,3,7,-1])
    return gt_bb


def some_many_few_several_half_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb,detidx=[11,12,13,21], lowerbound=[5,8,2,4], upperbound=[6,9,3,7]):
    for det, lowerb, upperb in zip(detidx, lowerbound, upperbound):
        # some 5-6, many 8-9, few 2-3, several 4-7
        alldet = np.argmax(input_cap[:,:25],axis=1) == det # captions with some
        allcountobj = np.argmax(input_cap[alldet, 25:], axis=1) < 10  # captions with countables
        # index of captions with some and countables
        nidx = np.arange(len(pred_bb))[alldet]
        nidx = nidx[allcountobj]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            presentobj = input_bb[n,:,4]>0
            objidx = np.argmax(input_bb[n,presentobj,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,presentobj,:4][objidx] #* 256

            if lowerb == -1 and upperb == -1:
                lb = up = len(allobjbb)//2
            else:
                lb, up = lowerb, upperb

            # logic: use predictions with score above 0.5 for any
            critanyidx = np.arange(max_bb)[pred_score[n]>0.5]
            predbb = pred_bb[n, critanyidx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]

            if not lb<=len(gtbb)<=up:
                if len(gtbb) < lb:
                    remainsoln = np.delete(np.arange(len(allobjbb)), gtbbsel)
                    idx = np.random.choice(remainsoln, lb-len(gtbb), replace=False)
                    gtbb = np.concatenate([gtbb, allobjbb[idx]],axis=0) # atleast 5 bounding boxes
                else:
                    descpred = np.argsort(pred_score[n])[::-1][:up]
                    gtbb = gtbb[descpred]

            assert (lb-1)<len(gtbb)<(up+1), print('some, many, few, several wrong gt')
            pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb


def a_an_either_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0,1,18]):
    # a, an, either
    for aidx in detidx:
        alla = np.argmax(input_cap[:,:25],axis=1) == aidx
        nidx = np.arange(len(pred_bb))[alla]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            presentobj = input_bb[n,:,4]>0
            objidx = np.argmax(input_bb[n,presentobj,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,presentobj,:4][objidx] #* 256

            # Logic: use top 1 prediction for a
            top1idx = np.argsort(pred_score[n])[::-1][:1]
            predbb = pred_bb[n, top1idx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]
            #assert len(gtbb) == 1
            if len(gtbb) !=1:
                print(gtbb)
            pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_gta/mod_gt')


def both_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=14):
    # both
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # Logic: use top 2 prediction for a
        top2idx = np.argsort(pred_score[n])[::-1][:2]
        predbb = pred_bb[n, top2idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 2
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb

def any_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3):
    # any
    allany = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allany]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: use predictions with score above 0.5 for any
        critanyidx = np.arange(max_bb)[pred_score[n]>0.5]
        predbb = pred_bb[n, critanyidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert 0 < len(gtbb) <= len(allobjbb)
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def this_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7):
    ## this
    allthis = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthis]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
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
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def that_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## that
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
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
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def these_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## these
    allthese = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthese]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
        this_thres = 170  # object must be bottom half of RGB image

        objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_that_bbs = allobjbb [objdist < this_thres]
        #target_bb = output_bb[n,:,:4]

        critthatidx = np.arange(max_bb)[pred_score[n]>0.5]
        predbb = pred_bb[n, critthatidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_that_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_that_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]

        # if len(correct_that_bbs) != len(predbb):
        #     print(np.argmax(input_bb[n, :, 5:], axis=1))
        #     print(len(correct_that_bbs), len(predbb))
        #     print('next')

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_that_bbs)), gtbbsel)
            if len(remainsoln)<1:
                print(gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_that_bbs[idx]],axis=0)
            #print('adding soln to these')

        assert len(gtbb) >1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def those_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10):
    ## those
    allthose = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(pred_bb))[allthose]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        center = np.array([128.0,256.0])
        this_thres = 128  # object must be bottom half of RGB image

        objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_that_bbs = allobjbb [objdist > this_thres]
        #target_bb = output_bb[n,:,:4]

        critthatidx = np.arange(max_bb)[pred_score[n]>0.5]
        predbb = pred_bb[n, critthatidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_that_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_that_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_that_bbs)), gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_that_bbs[idx]],axis=0)
            print('adding soln to those')
        assert len(gtbb) > 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_a_an_either_any_both_this_that_these_those_2')


if __name__ == '__main__':

    # gt_bb = generate_corrected_gt_json(gt_dir='../data_model/annotations/test_annotations.json', results_dir='../data_model/ns_results/test_results.json')


    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]
    #[pred_bb, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_only_v3', 1)
    #[pred_bb, pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_out_v3', 1)
    #[pred_bb,pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_det_v3',1)

    #create_output_txt(gdt=output_bb[:,:,:4], predt=pred_bb, confi=pred_score, directory='../data_model/ns_gt_bb/ori_gt')
    #create_output_txt(gdt=output_bb[:, :, :4], predt=pred_bb, confi=pred_score ,gd_cls=input_cap[:,:25],pred_cls=pred_cls,directory='../data_model/ns_cls_gt/ori_gt')

    #gt_bb = np.copy(output_bb[:,:,:4])  # modified ground truth bounding box labels

    #modgt_bb = main_change_gt_multiple_soln(gt_bb, input_bb, input_cap, pred_score, pred_bb)

    #create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score, directory='../data_model/ns_gt_bb/mod_gt')
    #create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:25],pred_cls=pred_cls,directory='../data_model/ns_cls_gt/mod_gt')



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


'''
def my_your_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[5,6]):
    # a, an, either
    for aidx in detidx:
        alla = np.argmax(input_cap[:,:25],axis=1) == aidx
        nidx = np.arange(len(pred_bb))[alla]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,:,:4][objidx] #* 256

            # tray coord
            trayidx = np.argmax(input_bb[n,:,5:],axis=1) == 15
            traycoord = input_bb[n,:,:4][trayidx]
            traydist = np.linalg.norm(center_coord(traycoord),axis=1)
            mytray = traycoord[np.argmax(traydist)]
            yourtray = traycoord[np.argmin(traydist)]

            # Logic: use top 1 prediction for a
            top1idx = np.argsort(pred_score[n])[::-1][:1]
            predbb = pred_bb[n, top1idx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]
            #assert len(gtbb) == 1
            if len(gtbb) !=1:
                print(gtbb)
            pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb
'''
