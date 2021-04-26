from ImageRecorder import ImageRecorder
import numpy as np
from PIL import Image, ImageDraw

# Load image and annotation files
img_filename = 'D:/Datasets/360-VOC2007/images/001579'
img = np.array(Image.open(img_filename+'.jpg'))
img_original = np.array(Image.open(img_filename+'.jpg'))
img_size = img.shape
annotation_path = 'D:/Datasets/360-VOC2007/annotations/001579.txt'

with open(annotation_path, "r") as r:
    img = Image.fromarray(img)
    annotations = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    for annotation in annotations[1:]:
        points_list = []
        label, theta, phi, fovw, fovh, difficult = annotation.split(' ')
        annotation = [np.radians(int(theta)), np.radians(int(phi)), int(fovw), int(fovh)]
        BFoV = ImageRecorder(img_size[1], img_size[0], view_angle_w=annotation[2],
                             view_angle_h=annotation[3])

        Px, Py = BFoV._sample_points(annotation[0], annotation[1])

        h, w = Px.shape

        for m in range(h):
            for n in range(w):
                if m == 0 or n == 0 or m == h - 1 or n == w - 1:
                    points_list.append((int(Px[m, n]) - 1, int(Py[m, n]) - 1, int(Px[m, n]) + 1, int(Py[m, n]) + 1))

        # Draw bounding fov
        draw = ImageDraw.Draw(img)
        for point in points_list:
            draw.ellipse(point, fill=None, outline=(255, 255, 0), width=2)

img.show('bfov')
# img.save('./a.jpg')
