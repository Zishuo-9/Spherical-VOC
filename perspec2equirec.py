import os
import cv2
from tqdm import tqdm
import lib.multi_Perspec2Equirec as m_P2E
import lib.Trans_annotations as B2F
import random

img_root = 'D:/Datasets'
dataset_name = 'VOCdevkit'
version = 'VOC2007'
input_dir = os.path.join(img_root, dataset_name, version, 'VOCtest2007')
output_dir = os.path.join(img_root, '360-'+version)
name_txt = os.path.join(input_dir, "ImageSets", "Main", 'test.txt')

with open(name_txt) as read:
    name_list = [line.strip().split('.')[0] for line in read.readlines()]

width = 1920
height = 960

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

json_dir = './pascal_voc_classes.json'

for img_name in tqdm(name_list):
    img_dir = os.path.join(input_dir, 'JPEGImages')
    xml_dir = os.path.join(input_dir, 'Annotations', img_name + '.xml')
    txt_dir = os.path.join(output_dir, 'annotations', img_name + '.txt')

    img_path = os.path.join(img_dir, img_name + ".jpg")

    VOC_input = cv2.imread(img_path, cv2.IMREAD_COLOR)
    VOCh, VOCw, _ = VOC_input.shape

    theta = int((random.random()*0.8 - 0.4) * 360)
    phi = int((random.random() - 0.5) * 180)
    fov = int(360 * VOCw / width)
    print(theta, phi)
    F_T_P = [fov, theta, phi]

    equ = m_P2E.Perspective([img_path], [F_T_P])
    img = equ.GetEquirec(height, width)

    transform = B2F.Bbox2Bfov(VOC_input, F_T_P, xml_dir, json_dir, height, width)
    bfov, bbox, labels, iscrowd = transform.readxml()

    index = 0
    with open(os.path.join(txt_dir), "w") as f:
        for i in range(len(bfov)):
            info1 = [str(theta), str(phi)]
            info2 = [str(j) for j in [labels[i], bfov[i][2], bfov[i][3], bfov[i][4], bfov[i][5], iscrowd[i]]]
            # print(info)
            if index == 0:
                f.write(" ".join(info1))
                f.write("\n" + " ".join(info2))
                index = 1
            else:
                f.write("\n" + " ".join(info2))

    cv2.imwrite(os.path.join(output_dir, 'images', img_name + '.jpg'), img)
