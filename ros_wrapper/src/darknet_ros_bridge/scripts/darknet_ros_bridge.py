#!/usr/bin/env python

import os.path
import copy
from functools import partial
import numpy as np
import string

import rospy
from geometry_msgs.msg import Polygon, Point32
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from wm_od_interface_msgs.msg import ira_dets, ira_det


def bounding_box_to_polygon(bbox):
    return Polygon(points=[
        Point32(x=bbox.xmin, y=bbox.ymin, z=0),
        Point32(x=bbox.xmax, y=bbox.ymin, z=0),
        Point32(x=bbox.xmax, y=bbox.ymax, z=0),
        Point32(x=bbox.xmin, y=bbox.ymax, z=0)])


def darknet_bounding_boxes_to_ira_dets(darket_bboxes):
    ira_detections = ira_dets()
    ira_detections.header = copy.deepcopy(darket_bboxes.header)
    # ira_detections.filename = "don't know"
    ira_detections.serial = darket_bboxes.header.seq # TODO Maybe?
    ira_detections.n_dets = len(darket_bboxes.bounding_boxes)
    for bbox in darket_bboxes.bounding_boxes:
        det = ira_det()
        # det.header = ira_detections.header
        det.obj_name = bbox.Class
        det.id = bbox.id
        # det.pose = ?
        det.bbox = bounding_box_to_polygon(bbox)
        det.confidence = bbox.probability
        ira_detections.dets.append(det)
    return ira_detections


def on_darknet_bbox_recv(ira_dets_pub, darket_bboxes):
    ira_dets_pub.publish(darknet_bounding_boxes_to_ira_dets(darket_bboxes))


def random_str(l):
    return ''.join(np.random.choice(list(string.lowercase), size=l))

def random_bbox():
    xmin, ymin, width, height = np.random.randint(0, 100, size=(4,))
    return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmin + width,
                       ymax=ymin+height, probability=np.random.rand(),
                       id=np.random.randint(100),
                       Class=random_str(6))

def random_darknet_bboxes():
    n_det = np.random.randint(10)
    bboxes = BoundingBoxes()
    bboxes.header.seq = np.random.randint(100)
    bboxes.header.stamp = rospy.Time(secs=np.random.randint(1000), nsecs=np.random.randint(1000))
    bboxes.header.frame_id = random_str(6)
    bboxes.image_header.seq = np.random.randint(100)
    bboxes.image_header.stamp = rospy.Time(secs=np.random.randint(1000), nsecs=np.random.randint(1000))
    bboxes.image_header.frame_id = random_str(6)
    for i in range(n_det):
        bboxes.bounding_boxes.append(random_bbox())
    return bboxes

def polygon_to_bounding_box(polygon):
    """
    >>> bbox = random_bbox()
    >>> bbox_rec = polygon_to_bounding_box(bounding_box_to_polygon(bbox))
    >>> bbox.xmin == bbox_rec.xmin
    True
    >>> bbox.ymin == bbox_rec.ymin
    True
    >>> bbox.xmax == bbox_rec.xmax
    True
    >>> bbox.ymax == bbox_rec.ymax
    True
    """
    ii64 = np.iinfo(type(BoundingBox().xmin))
    bbox = BoundingBox(xmin=ii64.max, ymin=ii64.max,
                       xmax=ii64.min, ymax=ii64.min)
    for pt in polygon.points:
        bbox.xmin = min(bbox.xmin, pt.x)
        bbox.ymin = min(bbox.ymin, pt.y)
        bbox.xmax = max(bbox.xmax, pt.x)
        bbox.ymax = max(bbox.ymax, pt.y)
    return bbox


def deepeq(obj1, obj2, comparabletypes=(str, bytes, int, float, tuple)):
    if not type(obj1) == type(obj2):
        return False

    if isinstance(obj1, comparabletypes) and isinstance(obj2, comparabletypes):
        return obj1 == obj2

    if isinstance(obj1, (tuple, list)) and isinstance(obj2, (tuple, list)):
        if len(obj1) != len(obj2):
            return False
        else:
            iseq = True
            for o1, o2 in zip(obj1, obj2):
                iseq = iseq and deepeq(o1, o2)
            return iseq
    elif hasattr(obj1, '__slots__') and hasattr(obj2, '__slots__'):
        iseq = True
        for k in obj1.__slots__:
            iseq = iseq and deepeq(getattr(obj1, k), getattr(obj2, k))
        return iseq
    else:
        raise RuntimeError("Dont know how to compare {0} and {1}".format(obj1, obj2))

def ira_dets_to_darknet_bounding_boxes(ira_detections):
    """
    >>> darknet_bboxes = random_darknet_bboxes()
    >>> darknet_bboxes_rec = ira_dets_to_darknet_bounding_boxes(
    ...                        darknet_bounding_boxes_to_ira_dets(
    ...                            darknet_bboxes))
    >>> deepeq(darknet_bboxes.header, darknet_bboxes_rec.header)
    True
    >>> deepeq(darknet_bboxes.bounding_boxes, darknet_bboxes_rec.bounding_boxes)
    True
    """
    bounding_boxes = BoundingBoxes()
    bounding_boxes.header = copy.deepcopy(ira_detections.header)
    bounding_boxes.image_header = copy.deepcopy(ira_detections.header)
    for det in ira_detections.dets:
        bbox = polygon_to_bounding_box(det.bbox)
        bbox.id = det.id
        bbox.Class = det.obj_name
        bbox.probability = det.confidence
        bounding_boxes.bounding_boxes.append(bbox)
    return bounding_boxes


def on_ira_dets_recv(darket_bbox_pub, ira_detections):
    darket_bbox_pub.publish(ira_dets_to_darknet_bounding_boxes(ira_detections))


def run_node():
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name, anonymous=True)
    ira_dets_pub = rospy.Publisher(
        "~ira_dets_out", ira_dets, queue_size=10)
    darknet_sub = rospy.Subscriber("~darknet_bounding_boxes_in",
                           BoundingBoxes,
                           partial(on_darknet_bbox_recv, ira_dets_pub))

    darknet_ros_pub = rospy.Publisher(
        "~darknet_bounding_boxes_out", BoundingBoxes, queue_size=10)
    ira_dets_sub = rospy.Subscriber("~ira_dets_in",
                           ira_dets,
                           partial(on_ira_dets_recv, darknet_ros_pub))

    rospy.spin()

if __name__ == '__main__':
    run_node()
