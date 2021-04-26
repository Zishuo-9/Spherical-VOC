import os
import sys
import cv2
import json
import numpy as np
from lxml import etree


class Bbox2Bfov:
    def __init__(self, img, F_T_P, xml_path, json_root, height, width):
        self.xml_path = xml_path

        self.h = height
        self.w = width

        self._img = img
        self._height, self._width, _ = self._img.shape

        self.wF, self.T, self.P = F_T_P
        self.hF = float(self._height) / self._width * self.wF

        self.json_file = open(json_root, 'r')
        self.class_dict = json.load(self.json_file)

        self.w_len = np.tan(np.radians(self.wF / 2.0))
        self.h_len = np.tan(np.radians(self.hF / 2.0))

        x, y = np.meshgrid(np.linspace(-180, 180, self.w), np.linspace(90, -90, self.h))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        self.xyz = np.stack((x_map, y_map, z_map), axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.T))  # Get rotation matrix
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.P))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        self.xyz = self.xyz.reshape([self.h * self.w, 3]).T
        self.xyz = np.dot(R2, self.xyz)
        self.xyz = np.dot(R1, self.xyz).T

        self.xyz = self.xyz.reshape([self.h, self.w, 3])

        self.inverse_mask = np.where(self.xyz[:, :, 0] > 0, 1, 0)

        self.xyz[:, :] = self.xyz[:, :] / np.repeat(self.xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    def readxml(self):
        with open(self.xml_path) as fid:
            xml_str = fid.read()

        xml = etree.fromstring(xml_str)
        data = self.parse_xml(xml)["annotation"]

        boxes = []
        bfov = []
        bbox = []
        points = []
        labels = []
        iscrowd = []

        for obj in data["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            xcen = int((xmin + xmax) / 2)
            ycen = int((ymin + ymax) / 2)

            boxes.append([xmin, xmax, ymin, ymax])
            points.append([xcen, ycen])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        for box, point in zip(boxes, points):
            anno1, anno2 = self.getequirec(box, point)

            bbox.append(anno2)
            bfov.append(anno1)

        assert len(bfov) == len(points)

        return bfov, bbox, labels, iscrowd

    def parse_xml(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])

        return {xml.tag: result}

    def getequirec(self, box, point):
        loc_map = np.where((( self.xyz[:, :, 1]+self.w_len)/2/self.w_len*self._width  > point[0]-2) &
                           (( self.xyz[:, :, 1]+self.w_len)/2/self.w_len*self._width  < point[0]+2) &
                           ((-self.xyz[:, :, 2]+self.h_len)/2/self.h_len*self._height > point[1]-2) &
                           ((-self.xyz[:, :, 2]+self.h_len)/2/self.h_len*self._height < point[1]+2), 1, 0)

        loc_map = loc_map * self.inverse_mask
        loc_array = np.where(loc_map == 1)

        if (loc_array[1].min() < 10) and (loc_array[1].max() > 1910):
            loc_mask = np.where(loc_array[1] < 960)
            loc_array[1][loc_mask] += 1920
            equx = np.mean(loc_array[1])
            if equx > 1920:
                equx -= 1920
        else:
            equx = np.mean(loc_array[1])
        equy = np.mean(loc_array[0])

        lon = (equx-self.w/2)/(self.w/2)*180
        lat = (self.h/2-equy)/(self.h/2)*90

        box_map = np.where((( self.xyz[:, :, 1]+self.w_len)/2/self.w_len*self._width  > box[0]) &
                           (( self.xyz[:, :, 1]+self.w_len)/2/self.w_len*self._width  < box[1]) &
                           ((-self.xyz[:, :, 2]+self.h_len)/2/self.h_len*self._height > box[2]) &
                           ((-self.xyz[:, :, 2]+self.h_len)/2/self.h_len*self._height < box[3]), 1, 0)
        box_map = box_map * self.inverse_mask
        box_array = np.where(box_map == 1)

        if (box_array[1].min() < 10) and (box_array[1].max() > 1910):
            equx_map = np.linspace(0, 1919, 1920).astype(int)
            locx_map = np.intersect1d(box_array[1], equx_map)

            if len(locx_map) == 1920:
                left  = np.min(box_array[1])
                right = np.max(box_array[1])
                up    = np.min(box_array[0])
                down  = np.max(box_array[0])

                bbox = [[left, up, right, down]]

            else:
                for i in range(len(locx_map)):
                    if (locx_map[i+1] - locx_map[i]) != 1:
                        edge = int((locx_map[i+1] - locx_map[i])/2)
                        break
                right_min = box_array[1][box_array[1] > edge].min()
                right_max = box_array[1][box_array[1] > edge].max()
                left_min = box_array[1][box_array[1] < edge].min()
                left_max = box_array[1][box_array[1] < edge].max()

                up, down = np.min(box_array[0]), np.max(box_array[0])

                bbox = [[left_min, up, left_max, down], [right_min, up, right_max, down]]

        else:
            left  = np.min(box_array[1])
            right = np.max(box_array[1])
            up    = np.min(box_array[0])
            down  = np.max(box_array[0])

            bbox = [[left, up, right, down]]

        box_map = box_map[:, :, np.newaxis]
        box_map = np.array(np.concatenate((box_map, box_map, box_map), axis=2), dtype='uint8')

        theta, phi = self.getperspec(box_map, lon, lat)

        return [int(equx), int(equy), int(lon), int(lat), int(theta), int(phi)], bbox

    def getperspec(self, equ, THETA, PHI, FOV=90, len=960):
        equ_cx = (self.w - 1) / 2.0
        equ_cy = (self.h - 1) / 2.0

        hFOV, wFOV = FOV, FOV

        height, width = len, len

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(equ, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        fov_map = np.where(persp[:, :, 0] == 1)

        theta1 = np.arctan((np.min(fov_map[1])-width/2)/(width/2)*np.tan(np.radians(wFOV/2)))*180/np.pi
        theta2 = np.arctan((np.max(fov_map[1])-width/2)/(width/2)*np.tan(np.radians(wFOV/2)))*180/np.pi

        phi1 = np.arctan((height/2-np.min(fov_map[0]))/(height/2)*np.tan(np.radians(hFOV/2)))*180/np.pi
        phi2 = np.arctan((height/2-np.max(fov_map[0]))/(height/2)*np.tan(np.radians(hFOV/2)))*180/np.pi

        return theta2-theta1, phi1-phi2

