"""
collect_dataset.py
"""

import os


def create_data_label_lists(dataset_folder, objs=range(1, 100+1), imgs=range(1, 310+1)):
    img_name_fmt = os.path.join(dataset_folder, 'obj{obj}', 'O{obj}_img{img}.jpg')
    data_list, label_list = [], []

    for obj in objs:
        for img in imgs:   # first inclusive, second exclusive
            data_list.append(img_name_fmt.format(obj=obj, img=img))
            label_list.append(obj-1)

    return data_list, label_list
