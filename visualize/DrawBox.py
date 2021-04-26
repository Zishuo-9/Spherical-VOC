from ImageRecorder import ImageRecorder
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json


# Load image and annotation files
img_filename = '../toBbox/7l0yq_bfov'
json_filename = '../annotations/7l0yq'
img = np.array(Image.open(img_filename+'.jpg'))
img_original = np.array(Image.open(img_filename+'.jpg'))
img_size = img.shape

json_file = json.load(open(json_filename+'.json'))
label = json_file['class']

box = []

for i in range(len(label)):
    annotation = json_file['boxes'][i][2:6]

    # Transform annotations to pixels in the image
    BFoV = ImageRecorder(img_size[1], img_size[0], view_angle_w=annotation[2], view_angle_h=annotation[3], long_side=img_size[1])
    Px, Py = BFoV._sample_points(annotation[0], annotation[1])

    # Calculate the boundary
    if json_file['boxes'][i][0] == 1049 or json_file['boxes'][i][0] == 170:
        if (Px.min() < 10) and (Px.max() > 1900):
            Px_right_min = int(Px[Px > Px.max()/2].min())
            Px_right_max = int(Px[Px > Px.max()/2].max())
            Px_left_min = int(Px[Px < Px.max()/2].min())
            Px_left_max = int(Px[Px < Px.max()/2].max())
            Py_min, Py_max = int(Py.min()), int(Py.max())
            if (Px_right_max != Px_right_min) and (Py_max != Py_min):
                box.append([Px_right_min, Py_min, Px_right_max, Py_max])
            if (Px_left_max != Px_left_min) and (Py_max != Py_min):
                box.append([Px_left_min, Py_min, Px_left_max, Py_max])
        else:
            Px_min, Px_max = int(Px.min()), int(Px.max())
            Py_min, Py_max = int(Py.min()), int(Py.max())
            if (Px_max != Px_min) and (Py_max != Py_min):
                box.append([Px_min, Py_min, Px_max, Py_max])

print(box)
# Draw bounding box
img = Image.fromarray(img)
draw = ImageDraw.Draw(img)
for i in range(len(box)):
    draw.rectangle((box[i][0], box[i][1], box[i][2], box[i][3]), fill=None, outline=(0, 0, 255), width=4)

# save image
img.show('test')
img.save(img_filename+'_bbox.jpg')
