from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

# custom colors
# COLORS = [(215,125,125),
#           ...]

class Debug:
    def __init__(self, dataset='train', size=600):
        self.dataset = dataset
        if dataset not in ['train', 'validation', 'test']:
            raise ValueError('dataset must be train one of {}, not \"{}\"'.format(['train', 'validation',
                                                                                   'test'], dataset))

        self.dir = 'Photos/Frags_{}/{}/'.format(size, dataset[0].upper()+dataset[1:])
        if dataset == 'validation':
            dataset = 'val'
        classes = pd.read_csv('Annotations/classes.csv', header=None, names=['class_', 'id_'])
        self.dict_classes = dict(zip(classes.class_, classes.id_))
        self.colors = [tuple(int(random.random() * 255) for _ in range(3)) for _ in range(len(classes['id_']))]

        data = pd.read_csv('Annotations/Frags_{}/{}_annotations.csv'.format(size, dataset), header=None,
                            names=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'label'])

        data.sort_values(['path'], inplace=True)

        self.data = data

        self.filenames = os.listdir(self.dir)

    def show(self, image_name):
        print(image_name)
        image = Image.open(self.dir+image_name)
        data = self.data.loc[self.data['path'] == self.dir+image_name].drop(columns=['path'])
        draw = ImageDraw.Draw(image)
        data = data.values.tolist()
        print('Frag has objects:', data[0] is not None)
        for row in data:
            *coords, label = row
            try:
                id = self.dict_classes[label]

                # for custom colors
                # color = COLORS[int(id)]

                # for random colors int(id)
                color = self.colors[int(id)]
            except:
                continue
            # draw.rectangle((int(row[0]), int(row[1]), int(row[2]), int(row[3])), outline=color, width=5)
            draw.rectangle(tuple(coords), outline=color, width=5)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)

    def show_random(self):
        image_name = list(self.filenames)[int(random.random() * len(self.filenames))]
        self.show(image_name)
