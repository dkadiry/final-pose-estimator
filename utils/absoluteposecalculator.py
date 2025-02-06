import csv
from scipy.spatial.transform import Rotation as R
import os
import numpy as np


def read_relative_poses_from_csv(relative_pose_file):
    relative_poses = []
    timestamps = []
    is_scaled = []
    objects_used = []
    with open(relative_pose_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            timestamps.append(row[0])
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            ox, oy, oz, ow = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            relative_poses.append((np.array([tx, ty, tz]), np.array([ox, oy, oz, ow])))
            is_scaled.append(row[8])
            objects_used.append(row[9])
    return timestamps, relative_poses, is_scaled, objects_used

def save_absolute_poses_to_csv(timestamps, absolute_poses, scale_flags, objects_used, output_file):
    with open(output_file, 'w') as f:
        f.write("pose_timestamp,tx,ty,tz,ox,oy,oz,ow,is_scaled,object_used\n")  # CSV header
        
        for i, (trans, quat) in enumerate(absolute_poses):
            if i == 0:
                timestamp = '1717696051836'
                is_scaled = 1
                object_used = 'None (Origin)'
            else:
                timestamp = timestamps[i-1]
                is_scaled = scale_flags[i-1]
                object_used = objects_used[i-1]

            f.write(f"{timestamp},{trans[0]},{trans[1]},{trans[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]},{is_scaled},{object_used}\n")

def compute_absolute_poses(relative_poses, initial_pose):
    absolute_poses = []
    
    # Start with the initial pose (Pose 1, typically the identity)
    absolute_poses.append(initial_pose)
    
    # Iterate over the relative poses to compute absolute poses
    for rel_pose in relative_poses:
        last_trans, last_quat = absolute_poses[-1]
        
        # Extract relative quaternion and translation from the relative pose
        rel_trans, rel_quat = rel_pose

        # Convert quaternions to Rotation objects for multiplication
        last_rotation = R.from_quat(last_quat)
        relative_rotation = R.from_quat(rel_quat)
        
        # Compute new absolute rotation by multiplying quaternions
        new_rotation = last_rotation * relative_rotation
        new_quat = new_rotation.as_quat()  # Convert back to quaternion
        
        # Rotate the relative translation by the last rotation
        rotated_translation = last_rotation.apply(rel_trans)
        
        # Compute new absolute translation
        new_trans = last_trans + rotated_translation
        
        # Append the new absolute pose
        absolute_poses.append((new_trans, new_quat))
    
    return absolute_poses

def main():
    pose_csv = os.path.join("Pose_Results", "final_raw_RFC_relative_poses_approach5_5.csv")
    init_pose = ((0, 0, 0),(0, 0, 0, 1))

    timestamps, rel_poses, scale_flag, obj_used = read_relative_poses_from_csv(pose_csv)
    abs_poses = compute_absolute_poses(rel_poses, init_pose)
    print ("Done computing absolute pose, saving to csv")

    out_file = os.path.join("Pose_Results", "final_raw_RFC_absolute_poses_approach5_5.csv")
    save_absolute_poses_to_csv(timestamps, abs_poses, scale_flag, obj_used, out_file)

    print("CSV file saved")

if __name__ == "__main__":
    main()