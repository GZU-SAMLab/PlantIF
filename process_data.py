import os
import random
from collections import defaultdict

from scipy import io as scio
import argparse


def convert_bird(data_root):
    images_txt = os.path.join(data_root, 'images.txt')
    train_val_txt = os.path.join(data_root, 'train_test_split.txt')
    labels_txt = os.path.join(data_root, 'image_class_labels.txt')

    id_name_dict = {}
    id_class_dict = {}
    id_train_val = {}
    with open(images_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, name = line.strip().split()
            id_name_dict[id] = name
            line = f.readline()

    with open(train_val_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, trainval = line.strip().split()
            id_train_val[id] = trainval
            line = f.readline()

    with open(labels_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            id, class_id = line.strip().split()
            id_class_dict[id] = int(class_id)
            line = f.readline()

    parts = os.path.join(data_root, 'parts/parts.txt')
    part_ids = {}
    with open(parts, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            each_line = line.strip().split()
            id = each_line[0]
            parts_name = " ".join(each_line[1:])
            part_ids[id] = parts_name
            line = f.readline()

    id_part_name = defaultdict(list)
    part_locs = os.path.join(data_root, 'parts/part_locs.txt')

    with open(part_locs, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            each_line = line.strip().split()
            id = each_line[0]
            part_id = each_line[1]
            id_part_name[id].append(part_ids[part_id])
            line = f.readline()

    attributes_file = os.path.join(data_root, 'attributes.txt')
    attributes_ids = {}

    with open(attributes_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            each_line = line.strip().split()
            id = each_line[0]
            attribute = each_line[1]
            attributes_ids[id] = attribute
            line = f.readline()
    train_txt = os.path.join(data_root, 'bird_train.txt')
    test_txt = os.path.join(data_root, 'bird_test.txt')
    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(test_txt):
        os.remove(test_txt)

    f1 = open(train_txt, 'a', encoding='utf-8')
    f2 = open(test_txt, 'a', encoding='utf-8')
    f1.write('id\tfile_path\tattributes\ttext\tlabel\n')
    f2.write('id\tfile_path\tattributes\ttext\tlabel\n')

    for id, trainval in id_train_val.items():
        temp_file_path = id_name_dict[id]
        temp_file_path = temp_file_path.replace('.jpg', '.txt')
        text_path = os.path.join(data_root, 'text', temp_file_path)
        text = open(text_path).readline().strip()

        if trainval == '1':

            f1.write(str(id) + '\t' + id_name_dict[id] + '\t' + attributes_ids[str(id_class_dict[id])] + '\t' + text + '\t' + str(id_class_dict[id] - 1)+'\n')
            # f1.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
        else:
            f2.write(str(id) + '\t' + id_name_dict[id] + '\t' + attributes_ids[str(id_class_dict[id])] + '\t' + text + '\t' + str(id_class_dict[id] - 1)+'\n')

            # f2.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
    f1.close()
    f2.close()


if __name__ == '__main__':
    convert_bird('../CUB_200_2011')
