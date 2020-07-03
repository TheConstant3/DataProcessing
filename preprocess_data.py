from PIL import Image, ImageDraw
from utils import *
from augment import *
import pandas as pd
import os
import json
import csv
import time
import argparse


def create_dataset(size_frag=600, intersection=100, rotate_all=True, DAFOD_augmentation=True,
                   laplacian_noise=True, vae_augmentation=False):
    # delete old data
    print('Deleting old files...')
    delete_old_frags_with_size(size_frag)
    print('Old files are deleted!')
    # creation directories for fragments
    try:
        os.mkdir('Photos/Frags_{}'.format(size_frag))
    except:
        print('Photos/Frags_{} is already created'.format(size_frag))
    for dir_name in ['Train', 'Test', 'Validation']:
        try:
            os.mkdir('Photos/Frags_{}/{}'.format(size_frag, dir_name))
        except:
            print('Photos/Frags_{}/ is already created'.format(size_frag, dir_name))

    files = os.listdir('Photos/Large/')
    all_annotations = []

    print('Fragments are creating...')
    for file in files:
        # crop each large image into fragments
        frags_info = create_frags(file, size_frag, intersection)
        if frags_info is None:
            return None
        # create annotation for every fragment
        annotations = create_annotations(file, frags_info)
        all_annotations.extend(annotations)
    print('Fragments are created!')

    classes = pd.read_csv('Annotations/classes.csv', header=None, names=['class', 'id'])
    all_annotations = pd.DataFrame(all_annotations, columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])

    # sort by paths
    all_annotations.sort_values(['path'], inplace=True)

    # if rotate_all=True start rotate each fragment to create new ones
    if rotate_all:
        print('Fragments are rotating...')
        all_annotations = rotate_images(all_annotations, size_frag)
        print('Fragments are rotated!')

    # split dataset
    train, val, test = split_dataset(all_annotations)

    train_dict = {}
    # if one of additional functions is necessary
    # convert annotations to dict type
    if vae_augmentation or DAFOD_augmentation or laplacian_noise:
        train_dict = get_dict_of_frags(train, classes)

    if vae_augmentation:
        print('Fragments are augmenting by vae...')
        train_dict = add_generate_objects(train_dict, size_frag)
        print('Fragments are augmented by vae!')

    if DAFOD_augmentation:
        print('Fragments are augmenting by DAFOD...')
        train_dict = do_DAFOD_augmentation(train_dict)
        print('Fragments are augmented by DAFOD!')

    if laplacian_noise:
        print('Fragments are augmenting by Laplacian noise...')
        train_dict = add_laplacian_noise(train_dict)
        print('Fragments are augmented by Laplacian noise!')

    # if annotations were converted to dict type
    # convert back to df
    if len(train_dict.keys()) != 0:
        print('Create list of annotations...')
        train = get_list_from_dict(train_dict, classes)
        train = pd.DataFrame(train, columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])
        print('List of annotations is created!')

    try:
        os.mkdir('Annotations/Frags_{}'.format(size_frag))
    except:
        print('Annotations/Frags_{} is already created'.format(size_frag))

    train = train.values.tolist()
    test = test.values.tolist()
    val = val.values.tolist()

    # convert to int values which are not None or str
    def convert_to_int(data):
        for i, row in enumerate(data):
            for j in range(1, 5):
                try:
                    data[i][j] = int(data[i][j])
                except:
                    data[i][j] = None
        return data

    train = convert_to_int(train)
    test = convert_to_int(test)
    val = convert_to_int(val)

    with open('Annotations/Frags_{}/train_annotations.csv'.format(size_frag), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train)

    with open('Annotations/Frags_{}/val_annotations.csv'.format(size_frag), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(val)

    with open('Annotations/Frags_{}/test_annotations.csv'.format(size_frag), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test)
    print('Annotations are created and saved!')


# creating fragments and return their coordinates on large photo
def create_frags(file, size, intersection):
    image = ''
    filepath = 'Photos/Large/' + file[:-4] + '.png'
    try:
        image = Image.open(filepath)
    except:
        print('{} cannot open!'.format(filepath))
    image_name = file[:-4]

    step = size - intersection

    # list for coordinates fragments on large photo
    positions_frags_on_photo = []

    # crop large image and get frags' coordinates on image
    for dir_name in ['Test', 'Train', 'Validation']:
        pos = crop_image_in_area(image, image_name, size, dir_name) if dir_name is not 'Train' \
                else crop_image_in_area(image, image_name, size, 'Train', step)
        positions_frags_on_photo.extend(pos)

    return positions_frags_on_photo


def crop_image_in_area(image, image_name, size, dir_name, step=None):
    if step is None:
        step = size
    width = image.size[0]
    height = image.size[1]
    # get coordinates area for cropping
    start_x, start_y, end_x, end_y = define_area(dir_name, width, height)

    num_of_frag = 0
    positions_frags_on_photo = []
    # flag for assign 'x' value '0'
    last_w = False
    # shift by width
    for x in range(start_x, end_x, step):
        if (x + size) > end_x:
            x = end_x - size
            last_w = True
        last_h = False

        # shift by height
        for y in range(start_y, end_y, step):
            if (y + size) > end_y:
                y = end_y - size
                last_h = True
            # crop fragment from image
            box = (x, y, x + size, y + size)
            frag = image.crop(box=box)

            path = 'Photos/Frags_{0}/{1}/{2}_{3}.png'.format(size, dir_name, image_name, num_of_frag)
            frag.save(path)

            # write fragment's position on large image
            positions_frags_on_photo.append([path] + [x, y] + [x + size, y + size])
            num_of_frag += 1

            if last_h:
                break
        if last_w:
            break

    return positions_frags_on_photo


def create_annotations(image_name, frags_info):
    annotations = []
    all_paths = set()
    frags_info = pd.DataFrame(frags_info, columns=['path', 'x1', 'y1', 'x2', 'y2'])
    classes = pd.read_csv('Annotations/classes.csv', header=None, names=['class', 'id'])

    # parse json
    with open('Annotations/labels.json', "r") as f_json:
        data = json.load(f_json)
        for img_data in data:
            filename = img_data['dataId'].split('\\')[-1]
            if filename == image_name:
                size_img = img_data['metadata']['image']
                w = size_img['width']
                h = size_img['height']
                for class_of_object in classes['class']:
                    for obj_data in img_data['label'][class_of_object]:
                        # convert relative coordinates to absolute
                        x_min, y_min, x_max, y_max = to_absolute(obj_data, w, h)

                        # write object data
                        filename = ''
                        obj_data = [filename] + [x_min, y_min, x_max, y_max] + [class_of_object]

                        оbject_on_frags = get_annotations_for_object(obj_data, frags_info)
                        annotations.extend(оbject_on_frags)

                        for frag_info in оbject_on_frags:
                            all_paths.add(frag_info[0])

                # write annotations for fragments without objects
                for path in frags_info['path']:
                    if path not in all_paths:
                        row = [path] + [None, None, None, None, None]
                        annotations.append(row)
                break
        return annotations


def split_dataset(annotations):

    train_i = annotations['path'].str.contains('Train')
    test_i = annotations['path'].str.contains('Test')
    val_i = annotations['path'].str.contains('Validation')

    train = annotations[train_i]
    test = annotations[test_i]
    val = annotations[val_i]

    return train, val, test


# get coordinates and class of object on fragments, which contain it
def get_annotations_for_object(object_on_large, frags_info):
    frags_info_list = frags_info.values.tolist()
    dict_obj_on_frag = {}
    for i, path in enumerate(frags_info['path']):
        # fragment's coordinates
        frag_on_large = frags_info_list[i]

        # if object's corners are in fragment
        if is_xymin_in_frag(object_on_large, frag_on_large) and \
                is_xymax_in_frag(object_on_large, frag_on_large):

            # write object's coordinates relative to fragment
            row = [path] + [object_on_large[1] - frag_on_large[1], object_on_large[2] - frag_on_large[2],
                            object_on_large[3] - frag_on_large[1], object_on_large[4] - frag_on_large[2]] \
                  + [object_on_large[5]]
            dict_obj_on_frag[i] = row

    object_on_frag = pd.DataFrame.from_dict(dict_obj_on_frag, orient='index').values.tolist()
    return object_on_frag


def get_list_from_dict(dict_of_frags, classes):
    annotations = []

    keys = list(dict_of_frags.keys())
    print(len(keys))
    for key in keys:
        for row in dict_of_frags[key]:
            try:
                class_name = classes[classes['id'] == row[-1]]['class'].values[0]
                annotations.append([key] + [int(i) for i in row[:-1]] + [class_name])
            except:
                annotations.append([key] + [None, None, None, None, None])

    return annotations


def get_dict_of_frags(data, classes):
    start = time.time()
    dict_of_frags = dict()
    paths = set(data['path'])
    for path in paths:
        ann = data.loc[data['path'] == path].drop(['path'], axis=1).values.tolist()
        for i, row in enumerate(ann):
            try:
                label = classes[classes['class'] == row[4]]['id'].values[0]
                ann[i][-1] = label
            except:
                ann = [[None, None, None, None, None]]
        dict_of_frags[path] = ann
    print('get_dict_of_frags {} sec'.format(time.time() - start))
    return dict_of_frags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess large image for object detection')
    parser.add_argument('size', type=int)
    parser.add_argument('intersection', type=int)
    parser.add_argument('-r', '--rotate_all', action="store_true")
    parser.add_argument('-d', '--dafod_augmentation', action="store_true")
    parser.add_argument('-l', '--laplacian_noise', action="store_true")
    parser.add_argument('-v', '--vae_augmentation', action="store_true")

    args = parser.parse_args()

    create_dataset(args.size, args.intersection, args.rotate_all, args.dafod_augmentation,
                    args.laplacian_noise, args.vae_augmentation)
