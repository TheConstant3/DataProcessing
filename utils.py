import numpy as np
import os


def correct_path(path, transform_info):
    filename = path.split('/')[-1]
    new_filename = filename.split('.')[0] + '_' + transform_info + '.' + filename.split('.')[1]
    path = path.replace(filename, new_filename)
    return path


def correct_data(new_data, img):
    all_data = []
    for row in new_data:
        try:
            all_data.append([img] + [int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])])
        except:
            all_data.append([img] + [None, None, None, None, None])

    return all_data


# return True if new bounding box intersect with one of other bounding boxes
def is_intersecting(new_bbox, bboxes):
    for bbox in bboxes:
        left = np.max([new_bbox[0], bbox[0]])
        top = np.max([new_bbox[1], bbox[1]])
        right = np.min([new_bbox[2], bbox[2]])
        bottom = np.min([new_bbox[3], bbox[3]])
        width = right - left
        height = bottom - top
        if width < 0 or height < 0:
            continue
        else:
            return True
    return False


def to_absolute(coords, w, h):
    x_min = int(coords['data']['min'][0] * w)
    y_min = int(coords['data']['min'][1] * h)
    x_max = int(coords['data']['max'][0] * w)
    y_max = int(coords['data']['max'][1] * h)
    return x_min, y_min, x_max, y_max


# return True if bottom left corner of bounding box is on fragment
def is_xymin_in_frag(bbox, frag_coords):
    if frag_coords[3] > bbox[1] >= frag_coords[1] \
            and frag_coords[4] > bbox[2] >= frag_coords[2]:
        return True
    return False


# return True if top right corner of bounding box is on fragment
def is_xymax_in_frag(bbox, frag_coords):
    if frag_coords[3] > bbox[3] >= frag_coords[1] \
            and frag_coords[4] > bbox[4] >= frag_coords[2]:
        return True
    return False


def delete_old_frags_with_size(size=600):
    test_path = 'Photos/Frags_{}/Test/'.format(size)
    val_path = 'Photos/Frags_{}/Validation/'.format(size)
    tr_path = 'Photos/Frags_{}/Train/'.format(size)
    try:
        for folder in [tr_path, val_path, test_path]:
            files = os.listdir(folder)
            for f in files:
                path = folder + f
                os.remove(path)
    except:
        pass


def show_amount_dataset_for_size(size=600):
    dirs = os.listdir('Photos/Frags_{}/Train/'.format(size))
    print(len(dirs), 'изображений в тренировочной выборке')
    dirs = os.listdir('Photos/Frags_{}/Validation/'.format(size))
    print(len(dirs), 'изображений в проверочной выборке')
    dirs = os.listdir('Photos/Frags_{}/Test/'.format(size))
    print(len(dirs), 'изображений в контрольной выборке')


# define area of large photo, then will cropped
# top left quarter of photo - test data
# top right quarter of photo - validation data
# bottom half of photo - train data
def define_area(dir_name, width, height):
    if dir_name == 'Test':
        start_x = 0
        start_y = 0
        end_x = width // 2
        end_y = height // 2
    elif dir_name == 'Validation':
        start_x = width // 2
        start_y = 0
        end_x = width
        end_y = height // 2
    else:
        start_x = 0
        start_y = height // 2
        end_x = width
        end_y = height
    return start_x, start_y, end_x, end_y
