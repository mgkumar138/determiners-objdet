import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# https://github.com/Cartucho/mAP

def custom_loss(y_true, y_pred):
    bc_loss = tf.keras.metrics.binary_crossentropy(y_true[:,:20], y_pred[:,:20], from_logits=False)
    cat_loss = tf.keras.metrics.categorical_crossentropy(y_true[:,20:], y_pred[:,20:], from_logits=False)
    loss = tf.reduce_mean(bc_loss + cat_loss)
    return loss

def create_output_txt(gdt, predt, confi, directory, gd_cls=None,pred_cls=None):
    # gd: <class_name> <left> <top> <right> <bottom> [<difficult>]
    # pred : <class_name> <confidence> <left> <top> <right> <bottom>
    if gd_cls is not None:
        # determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
        #                "few", "both", "neither", "little", "much", "either", "our"]
        determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                       "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                       "the"]
        gd_cls_id = np.argmax(gd_cls,axis=1)
        pred_cls_id = np.argmax(pred_cls,axis=1)

    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+'/ground-truth')
        os.makedirs(directory + '/detection-ns_results')

    N = gdt.shape[0]
    for n in range(N):

        posidx = np.arange(gdt.shape[1])[np.any(gdt[n,:,:4]>0,axis=1)]
        with open('./{}/ground-truth/imcap_{}.txt'.format(directory,n), 'w') as fh:
            for i in posidx:
                if gd_cls is not None:
                    cl = determiners[gd_cls_id[n]]
                else:
                    cl = 'none'
                bb = xywh_to_coord(gdt[n,i,:4])
                fh.write('{} {} {} {} {}\n'.format(cl, bb[0],bb[1],bb[2],bb[3]))

        posposidx = np.any(predt[n,:,:4]>0,axis=1)*(confi[n]>0.5)
        posidx = np.arange(predt.shape[1])[posposidx]
        with open('./{}/detection-ns_results/imcap_{}.txt'.format(directory,n), 'w') as fh:
            for i in posidx:
                if pred_cls is not None:
                    cl = determiners[pred_cls_id[n]]
                    if cl == "neither":
                        print('neither')
                else:
                    cl = 'none'
                bb = xywh_to_coord(predt[n,i,:4])
                confidence = confi[n,i]
                fh.write('{} {} {} {} {} {}\n'.format(cl, confidence, bb[0],bb[1],bb[2],bb[3]))



def generate_img_caption_bb_mask(dir='../dataset_dir', show_example=True):
    import json
    from PIL import Image

    coco_annotation_input_file_path = dir+"/annotations/train_input_labels.json"
    coco_annotation_output_file_path = dir+"/annotations/train_output_labels.json"

    # read json
    with open(coco_annotation_input_file_path, 'r') as f:
        input_dataset = json.loads(f.read())

    with open(coco_annotation_output_file_path, 'r') as f:
        output_dataset = json.loads(f.read())

    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our"]

    images = input_dataset["images"]
    categories = input_dataset["categories"]
    input_annotations = input_dataset["annotations"]
    input_phrases = input_dataset["phrase_annotations"]
    output_annotations = output_dataset["annotations"]

    images_array = []
    segmentation_mask_array = []
    output_bboxes_array = []
    input_bboxes_array = []
    input_determiners_array = []
    captions_array = []

    n_categories = len(categories)
    max_bboxes = 20

    for i in images:
        filename = i["filename"]
        filepath = dir + "/" + filename
        img = Image.open(filepath)
        img = np.asarray(img)
        images_array.append(img)

        segmentation_filename = filename[:-8] + "_segmentation.png"
        segmentation_filename = "segmentations\\" + segmentation_filename[7:]
        segmentation_filepath = dir + "/" + segmentation_filename
        segmentation_img = Image.open(segmentation_filepath)
        segmentation_img = segmentation_img.convert("L")
        segmentation_img = segmentation_img.point(lambda p: 0 if p == 255 else 1)
        segmentation_img = np.asarray(segmentation_img)
        segmentation_mask_array.append(segmentation_img)

        image_id = i["id"]
        output_bboxes_array.append([])
        input_bboxes_array.append([])

    for ann in output_annotations:
        one_hot = [0 for i in range(n_categories)]
        one_hot[ann["category_id"]] = 1
        output_bboxes_array[ann["image_id"]].append(ann["bbox"] + [1] + one_hot)
        if ann["bbox"][2] == 0 or ann["bbox"][3] == 0:
            print("wrong", ann["category_id"], ann["bbox"], ann["image_id"])

    for i, bboxes in enumerate(output_bboxes_array):
        count = len(bboxes)
        one_hot = [0 for i in range(n_categories)]
        for j in range(count, max_bboxes):
            output_bboxes_array[i].append([0, 0, 0, 0] + [0] + one_hot)

    for ann in input_annotations:
        one_hot = [0 for i in range(n_categories)]
        one_hot[ann["category_id"]] = 1
        if ann["bbox"][2] == 0 or ann["bbox"][3] == 0:
            print(ann["bbox"], ann["category_id"])
        input_bboxes_array[ann["image_id"]].append(ann["bbox"] + [1] + one_hot)

    for i, bboxes in enumerate(input_bboxes_array):
        count = len(bboxes)
        one_hot = [0 for i in range(n_categories)]
        for j in range(count, max_bboxes):
            input_bboxes_array[i].append([0, 0, 0, 0] + [0] + one_hot)

    # save captions
    for p in input_phrases:
        captions_array.append(p["caption"])
        det = p["caption"].split()[0]
        noun = p["caption"].split()[1]
        determiners.index(det)
        noun_id = 0
        for cat in categories:
            if cat["name"] == noun:
                noun_id = cat["id"]
        det_one_hot = [0 for i in range(len(determiners))]
        det_one_hot[determiners.index(det)] = 1
        noun_one_hot = [0 for i in range(len(categories))]
        noun_one_hot[noun_id] = 1
        input_determiners_array.append(det_one_hot + noun_one_hot)

    # convert to numpy arrays
    image_matrix = np.stack(images_array, axis=0)
    segmentation_mask_matrix = np.stack(segmentation_mask_array, axis=0)
    captions_matrix = np.array(captions_array)
    output_bboxes_matrix = np.stack(output_bboxes_array, axis=0)
    input_bboxes_matrix = np.stack(input_bboxes_array, axis=0)
    input_determiners_matrix = np.stack(input_determiners_array, axis=0)

    print("Image matrix shape: ", image_matrix.shape)
    print("Phrase matrix shape: ", len(captions_array))
    print("Output bboxes matrix shape: ", output_bboxes_matrix.shape)
    print("ObjDet bboxes matrix shape: ", input_bboxes_matrix.shape)
    print("ObjDet Segmentation mask shape", segmentation_mask_matrix.shape)
    print("WordEmbed determiners matrix shape: ", input_determiners_matrix.shape)

    if show_example:
        import matplotlib.patches as patches
        f,ax = plt.subplots(3,4)
        ridx = np.random.choice(np.arange(len(captions_array)), 3,replace=False)
        for i in range(3):
            j = ridx[i]
            ax[i,0].imshow(image_matrix[ridx[i]])
            ax[i,0].set_title(captions_array[ridx[i]], fontsize=10)
            ax[i,0].set_xticks([])
            ax[i, 0].set_yticks([])

            ax[i,1].imshow(image_matrix[ridx[i]])
            ax[i,1].set_xticks([])
            ax[i, 1].set_yticks([])
            for bb in range(20):
                allbb = input_bboxes_matrix[j,bb]
                truerect = patches.Rectangle((allbb[0], allbb[1]), allbb[2], allbb[3],linewidth=2, edgecolor='g', facecolor='none')
                ax[i,1].add_patch(truerect)

            ax[i,2].imshow(image_matrix[ridx[i]])
            ax[i,2].set_xticks([])
            ax[i, 2].set_yticks([])
            for bb in range(20):
                allbb = output_bboxes_matrix[j,bb]
                truerect = patches.Rectangle((allbb[0], allbb[1]), allbb[2], allbb[3],linewidth=2, edgecolor='r', facecolor='none')
                ax[i,2].add_patch(truerect)

            ax[i,3].imshow(segmentation_mask_matrix[ridx[i]])
            ax[i,3].set_xticks([])
            ax[i, 3].set_yticks([])
            #plt.colorbar(im,fraction=0.046, pad=0.04)
        #plt.tight_layout()
        plt.show()
    saveload('save','../dataset_dir/img_cap_bb_mask_matrix_{}'.format(len(images)),
             [image_matrix, captions_matrix, output_bboxes_matrix,
              input_bboxes_matrix, segmentation_mask_matrix,input_determiners_matrix])
    #return image_matrix, captions_array,input_bboxes_matrix, output_bboxes_matrix, segmentation_mask_matrix


def train_val_test_split(images, captions, targets, objdet, sentemb, splitratio=[0.7,0.15,0.15], seed=0):
    np.random.seed(seed)
    N = len(targets)
    allidx = np.arange(N)
    testidx = np.random.choice(allidx, int(N*splitratio[2]), replace=False)
    validx = np.random.choice(np.delete(allidx,testidx), int(N*splitratio[1]), replace=False)
    trainidx = np.delete(allidx,np.concatenate([testidx,validx]))

    return [images[trainidx], captions[trainidx], targets[trainidx], objdet[trainidx], sentemb[trainidx]], \
           [images[validx], captions[validx], targets[validx], objdet[validx], sentemb[validx]],\
           [images[testidx], captions[testidx], targets[testidx], objdet[testidx], sentemb[testidx]],



def get_USE_model():
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    text_model = hub.load(module_url)
    return text_model


def saveload(opt, name, variblelist):
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var

def xywh_to_coord(boxes):
    return np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], axis=-1)


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
    # return the intersection over union value
    return iou

def compute_all_ious(gtbb,predbb):
    allious = np.zeros([len(gtbb), len(predbb)])
    for i in range(len(gtbb)):
        for j in range(len(predbb)):
            allious[i,j] = bb_intersection_over_union(xywh_to_coord(gtbb[i]), xywh_to_coord(predbb[j]))
    return allious

def center_coord(boxes):
    return np.concatenate([boxes[..., :2] + boxes[..., 2:]/2], axis=-1)

def base_coord(boxes):
    return np.concatenate([boxes[..., :2] + boxes[..., 2:]/2], axis=-1)