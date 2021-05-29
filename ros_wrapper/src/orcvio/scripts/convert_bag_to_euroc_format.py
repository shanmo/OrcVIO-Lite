#!/usr/bin/env python
"""
Example usage is with medfield_ft8_2020-11-06-16-12-43.bag
https://www.dropbox.com/sh/rvrlsqrcxyk8glo/AAAUHzQ8bnJqWPIU-4g7t1bZa/phoenix_stack/2020-11-06%20Medfield?dl=1

Required topics in the bagfile:

    $ rosbag info MIT-Jackal-Phoenix-Medfield/medfield_ft8_2020-11-06-16-12-43.bag | grep -E 'image_raw|forward/imu|microstrain/data'
                /acl_jackal/forward/color/image_raw/compressed                 2759 msgs    : sensor_msgs/CompressedImage               
                /acl_jackal/forward/imu                                       18269 msgs    : sensor_msgs/Imu                           
                /acl_jackal/microstrain/data                                  11501 msgs    : sensor_msgs/Imu                           


The bagfile has compressed image convert to uncompressed images using image_transport republish. Use ../launch/uncompress_image_bag.launch

The following output should be expected

   $ rosbag info MIT-Jackal-Phoenix-Medfield/medfield_ft8_2020-11-06-16-12-43-uncompressed.bag 
path:        MIT-Jackal-Phoenix-Medfield/medfield_ft8_2020-11-06-16-12-43-uncompressed.bag
version:     2.0
duration:    1:31s (91s)
start:       Mar 25 2021 14:27:58.38 (1616707678.38)
end:         Mar 25 2021 14:29:30.37 (1616707770.37)
size:        2.4 GB
messages:    41844
compression: none [2752/2752 chunks]
types:       sensor_msgs/Image  [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/Imu    [6a62c6daae103f4ff57a132d6f95cec2]
             tf2_msgs/TFMessage [94810edda583a504dfda3829e70d7eec]
topics:      /tf                                            9332 msgs    : tf2_msgs/TFMessage
             acl_jackal/forward/color/image_uncompressed    2751 msgs    : sensor_msgs/Image 
             acl_jackal/forward/imu                        18263 msgs    : sensor_msgs/Imu   
             acl_jackal/microstrain/data                   11498 msgs    : sensor_msgs/Imu


The this bag can be converted into euroc format

   $ python convert_bag_to_euroc_format.py MIT-Jackal-Phoenix-Medfield/medfield_ft8_2020-11-06-16-12-43-uncompressed.bag

Then data can be explored in the files like:
   $ head MIT-Jackal-Phoenix-Medfield/medfield_ft8_2020-11-06-16-12-43_euroc_format/acl_jackal/forward-imu/data.csv
"""
import csv
import os.path as osp
import os

import rosbag
import rospy
import matplotlib.pyplot as plt
import transforms3d
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

def ensuredirs(filepath):
    filedir = osp.dirname(filepath)
    if not osp.exists(filedir):
        os.makedirs(filedir)
    return filepath

def extract_imu_from_bag(bagfile,
                         euroc_dir='euroc_format/acl_jackal/forward-imu',
                         imu_topic='acl_jackal/forward/imu'):
    with open(ensuredirs(osp.join(euroc_dir, 'data.csv')), 'w') as datacsv:
        csvwriter = csv.writer(datacsv, quoting=csv.QUOTE_MINIMAL)
        datacsv.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for topic, msg, t in rosbag.Bag(bagfile).read_messages([imu_topic]):
            w_RS_S = msg.angular_velocity
            a_RS_S = msg.linear_acceleration
            csvwriter.writerow([msg.header.stamp.to_nsec(),
                                w_RS_S.x, w_RS_S.y, w_RS_S.z,
                                a_RS_S.x, a_RS_S.y, a_RS_S.z])

def extract_cam_from_bag(bagfile,
                         euroc_dir='euroc_format/acl_jackal/forward-cam',
                         cam0_topic='acl_jackal/forward/color/image_uncompressed'):
    bridge = CvBridge()
    with open(ensuredirs(osp.join(euroc_dir, 'data.csv')), 'w') as datacsv:
        csvwriter = csv.writer(datacsv, quoting=csv.QUOTE_MINIMAL)
        datacsv.write('#timestamp [ns], filename\n')
        for topic, msg, t in rosbag.Bag(bagfile).read_messages([cam0_topic]):
            imgname = '%d.png' % msg.header.stamp.to_nsec()
            csvwriter.writerow([msg.header.stamp.to_nsec(), imgname])
            imgfilepath = osp.join(euroc_dir, 'data', imgname)
            if not osp.exists(imgfilepath):
                frame = bridge.imgmsg_to_cv2(msg, "bgr8")
                plt.imsave(ensuredirs(imgfilepath), frame)

def extract_tf_map_to_base_from_bag(bagfile,
                                    euroc_dir='euroc_format/acl_jackal/pos_estimate0',
                                    delimiter=' '):
    tfBuffer = tf2_ros.Buffer()
    max_error_count = 10
    error_count = 0
    with open(ensuredirs(osp.join(euroc_dir, 'data.txt')), 'w') as datacsv:
        csvwriter = csv.writer(datacsv, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        datacsv.write('#' + delimiter.join('timestamp(s) tx ty tz qx qy qz qw'.split()))
        datacsv.write('\n')
        for topic, msg, t in rosbag.Bag(bagfile).read_messages(['/tf', '/tf_static']):
            isstatic = (topic == '/tf_static')
            for trans in msg.transforms:
                if isstatic:
                    tfBuffer.set_transform_static(trans, "bag")
                else:
                    tfBuffer.set_transform(trans, "bag")
            try:
                trans = tfBuffer.lookup_transform('acl_jackal/base', 'acl_jackal/map', rospy.Time(0))
            except tf2_ros.LookupException as e:
                print(str(e))
                if error_count < max_error_count:
                    print("Ignoring")
                    error_count += 1
                    continue
                else:
                    raise
            except tf2_ros.ExtrapolationException as e:
                print(str(e))
                if error_count < max_error_count:
                    print("Ignoring")
                    error_count += 1
                    continue
                else:
                    raise
            except tf2_ros.ConnectivityException as e:
                print(str(e))
                if error_count < max_error_count:
                    print("Ignoring")
                    error_count += 1
                    continue
                else:
                    raise

            p = trans.transform.translation
            q = trans.transform.rotation
            csvwriter.writerow([trans.header.stamp.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w])


def convert_bag_to_euroc(bagfile):
    euroc_format_root = osp.splitext(bagfile)[0] + '_euroc_format'
    extract_cam_from_bag(
        bagfile,
        euroc_dir=osp.join(euroc_format_root, 'acl_jackal', 'forward-infra1'),
        cam0_topic='acl_jackal/forward/infra1/image_uncompressed')
    extract_imu_from_bag(bagfile,
        euroc_dir=osp.join(euroc_format_root, 'acl_jackal', 'forward-imu'),
        imu_topic='acl_jackal/forward/imu')
    extract_imu_from_bag(
        bagfile,
        euroc_dir=osp.join(euroc_format_root, 'acl_jackal', 'microstrain'),
        imu_topic='acl_jackal/microstrain/data')
    extract_tf_map_to_base_from_bag(
        bagfile,
        euroc_dir=osp.join(euroc_format_root, 'acl_jackal', 'pos_estimate0'))

if __name__ == '__main__':
    import sys
    bagfile = sys.argv[1]
    convert_bag_to_euroc(bagfile)
