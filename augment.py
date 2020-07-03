from DataAugmentationForObjectDetection.data_aug.data_aug import *
from DataAugmentationForObjectDetection.data_aug.bbox_util import *
# from DataProcessing.preprocess_data import *
from utils import *
from skimage.transform import resize
from PIL import Image

import pandas as pd

# import keras
import cv2


# region rotate
# calculate new object's coordinates after rotate on '90', '180' and '270' degrees
def rotate_bbox(row, angle, size, path):
    if row[1] is None:
        bbox = [path] + [None, None, None, None, None]
    elif angle == '90':
        y_min = size - row[2]
        y_max = size - row[0]
        x_min = row[1]
        x_max = row[3]
        bbox = [path] + [x_min, y_min, x_max, y_max] + [row[4]]
    elif angle == '180':
        y_min = size - row[3]
        y_max = size - row[1]
        x_min = size - row[2]
        x_max = size - row[0]
        bbox = [path] + [x_min, y_min, x_max, y_max] + [row[4]]
    elif angle == '270':
        y_min = row[0]
        y_max = row[2]
        x_min = size - row[3]
        x_max = size - row[1]
        bbox = [path] + [x_min, y_min, x_max, y_max] + [row[4]]
    return bbox


# rotate fragments and get new annotations
def rotate_images(annotations, size=600):
    dict_annotations = dict()
    i = 0
    paths = set(annotations['path'])
    for path_frag in paths:
        img = Image.open(path_frag)
        bboxes = annotations.loc[annotations['path'] == path_frag].drop(['path'], axis=1).values.tolist()

        # rotate fragment
        for angle in ['90', '180', '270']:
            new_path = correct_path(path_frag, angle)
            rot_img = img.rotate(int(angle))
            rot_img.save(new_path)

            # rotate bounding boxes
            for bbox in bboxes:
                i += 1
                bbox = rotate_bbox(bbox, angle, size, new_path)
                dict_annotations[i] = bbox

    df = pd.DataFrame.from_dict(dict_annotations, orient='index', columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
    return annotations.append(df)
# endregion rotate


# augmentation with DataAugmentationForObjectDetection
# https://github.com/Paperspace/DataAugmentationForObjectDetection
# region DataAugmentationForObjectDetection

# optional
dict_of_methods = {RandomHorizontalFlip(1): '_horflip',
                   RandomHSV(60, 20, 10): '_hsv',
                   RandomRotate(45): '_rotate'}


def transform(method, img, frag, bboxes, is_empty):
    method_name = dict_of_methods[method]
    # if fragment without objects
    if is_empty:
        empty_bbox = np.array([[1, 1, 2, 2, 1]], dtype=np.float64)
        img_, bboxes_ = method(img, empty_bbox)
        bboxes_ = np.array([[None,None,None,None,None]])
    else:
        img_, bboxes_ = method(img, bboxes)
    frag_ = correct_path(frag, method_name)
    cv2.imwrite(frag_, cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))

    return correct_data(bboxes_, frag_)


def do_DAFOD_augmentation(dict_of_frags):
    keys = list(dict_of_frags.keys())
    for frag in keys:
        img = cv2.imread(frag)[:, :, ::-1]
        bboxes = dict_of_frags[frag]
        bboxes = np.array(bboxes, dtype=np.float64)
        is_empty = np.isnan(bboxes[:, 0])

        is_empty = is_empty[0]

        for method in dict_of_methods.keys():
            new_annotations = transform(method, img.copy(), frag, bboxes.copy(), is_empty)
            frag_name = new_annotations[0][0]

            # write new annotations in dict
            dict_of_frags[frag_name] = [new_annotations[0][1:]]
            for row in new_annotations[1:]:
                dict_of_frags[frag_name].append(row[1:])

    return dict_of_frags
# endregion DataAugmentationForObjectDetection


# create fragments with noise
def add_laplacian_noise(dict_of_frags):
    frag_names = list(dict_of_frags.keys())
    for frag in frag_names:
        # crate new frag with noise
        img = cv2.imread(frag)[:, :, ::-1]
        img_ = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
        img_ = img + img_
        frag_ = correct_path(frag, 'laplacian')
        cv2.imwrite(frag_, cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))

        # write new annotations
        bboxes = dict_of_frags[frag]
        bboxes = np.array(bboxes, dtype=np.float64)
        data = correct_data(bboxes, frag_)
        dict_of_frags[frag_] = [data[0][1:]]
        for row in data[1:]:
            dict_of_frags[data[0][0]].append(row[1:])
    return dict_of_frags


# generate object image and add it to fragment
def add_generate_objects(dict_of_frags, size_frag):
    # optional
    max_new_objects = 20
    max_side_size = 64
    amount_of_classes = 4

    dict_of_vaes = {x: keras.models.load_model('vaes/vae_class_{0}.h5'.format(x), \
                                               compile=False).layers[-1] for x in range(amount_of_classes)}
    frag_names = list(dict_of_frags.keys())
    for frag in frag_names:
        img = cv2.imread(frag)[:, :, ::-1]

        # get bounding boxes
        bboxes = dict_of_frags[frag]
        bboxes = np.array(bboxes, dtype=np.float64)
        empty_frag = np.isnan(bboxes[:, 0])

        has_new_objects = False
        for i in range(max_new_objects):
            # generate coordinates and size
            x = int(random.random() * (size_frag - max_side_size))
            y = int(random.random() * (size_frag - max_side_size))
            scale = random.random() * 0.5 + 0.5
            l = int(max_side_size * scale)
            new_bbox_coords = np.array([x, y, x + l, y + l])
            empty_frag = empty_frag[0]

            # new bbox should not intersect with old bboxes
            if is_intersecting(new_bbox_coords, bboxes[:, :-1]) is False or empty_frag:
                has_new_objects = True
                new_label = int(random.random() * amount_of_classes)
                generator = dict_of_vaes[new_label]

                # generate object from random point of latent space
                z = np.array([[np.random.random(), np.random.random()]])
                new_object = generator.predict(z)
                new_object = new_object[0].reshape((max_side_size, max_side_size, 3)) * 255
                new_object = resize(new_object, (l, l))

                # add new object to fragment
                img[int(new_bbox_coords[1]):int(new_bbox_coords[3]),
                    int(new_bbox_coords[0]):int(new_bbox_coords[2]), :] = new_object
                bbox = np.array(new_bbox_coords.tolist() + [new_label])

                # if frag was empty
                if empty_frag:
                    bboxes = np.array([bbox])
                    empty_frag = False
                else:
                    bboxes = np.vstack((bboxes, bbox))

        if has_new_objects:
            frag_ = correct_path(frag, 'vae')
            cv2.imwrite(frag_, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            bboxes = np.array(bboxes, dtype=np.float64)
            data = correct_data(bboxes, frag_)
            dict_of_frags[frag_] = [data[0][1:]]
            for row in data[1:]:
                dict_of_frags[frag_].append(row[1:])

    return dict_of_frags
