#!/usr/bin/env python
import rosbag
from tf_bag import BagTfTransformer

import numpy as np
import os 
from pyquaternion import Quaternion

def save_pose(timestamps, positions, rotations, result_dir):
    """
    save positions, rotations of the trajectory
    :param timestamps: list of timestamps
    :param positions: list of positions
    :param rotations: list of rotations
    :param result_dir: directory of results
    """

    with open(result_dir, "w") as f:

        for state_id in range(0, len(timestamps)):

            timestamp = timestamps[state_id]
            s1 = " ".join(map(str, positions[state_id]))
            q = rotations[state_id]
            s2 = " ".join(map(str, q))

            f.write("%s %s %s\n" % (timestamp, s1, s2))

if __name__ == "__main__":

    root_dir = "/media/erl/disk1/orcvio/mit_rosbags/phoenix_stack/"
    input_bag = root_dir + "medfield_ft5_2020-11-06-15-44-24.bag"
    result_dir = "/home/erl/Workspace/orcvio-lite/cache/stamped_groundtruth.txt"

    bag_transformer = BagTfTransformer(input_bag)

    # frame1_id = 'acl_jackal/forward_color_optical_frame'
    # frame1_id = 'acl_jackal/base'
    # frame2_id = 'acl_jackal/map'

    frame1_id = 'acl_jackal/map'
    frame2_id = 'acl_jackal/base'

    timestamps = []
    positions = []
    rotations = []

    for topic, msg, time in rosbag.Bag(input_bag).read_messages(): 
        timestamps.append(time.to_sec())
        translation, quaternion = bag_transformer.lookupTransform(frame1_id, frame2_id, time)
        positions.append(translation)
        rotations.append(quaternion)
    
    # bag_transformer.plotTranslation(frame1_id, frame2_id)
    save_pose(timestamps, positions, rotations, result_dir)