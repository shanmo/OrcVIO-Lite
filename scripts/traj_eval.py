import numpy as np
import os, glob, sys
import os.path as path
import shutil

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) +
    "/rpg_trajectory_evaluation/" + "src/rpg_trajectory_evaluation")

from trajectory import Trajectory

class TrajEval():
    """
    this class evaluates the trajectory wrt groundtruth
    """

    def __init__(self, result_dir):

        # ref http://www.cvlibs.net/datasets/kitti/eval_odometry.php
        self.subtraj_lengths = [x * 100 for x in range(1, 9)]

        self.result_dir = result_dir

    def evaluate_and_plot(self):
        """
        evaluate the poses and show the plot
        """

        # <result_folder> should contain the groundtruth, trajectory estimate and
        # optionally the evaluation configuration as mentioned above.

        # remove old results
        folder_name = self.result_dir + 'plots/'
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)

        folder_name = self.result_dir + 'saved_results/'
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)

        evo_cmd = 'python ' + './rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py ' \
                  + self.result_dir

        os.system(evo_cmd)

if __name__ == "__main__":

    result_dir = "/home/vdhiman/.cache/orcvio_cpp"
    TE = TrajEval(result_dir)
    TE.evaluate_and_plot()


