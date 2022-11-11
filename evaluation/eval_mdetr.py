from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from evaluation.eval_det import generate_corrected_gt_json

annFile = './annotations/gt_test_annotations.json'
cocoGt = COCO(annFile)

resFile = './mdetr_results/results_mdetr.json'
cocoDt = cocoGt.loadRes(resFile)

annType = "bbox"

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

#
generate_corrected_gt_json(gt_dir=annFile, results_dir=resFile, max_bboxes=100)
#
modannFile = './annotations/mod_test_annotations.json'
modcocoGt = COCO(modannFile)

print('After correcting annotations')
cocoEval = COCOeval(modcocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()