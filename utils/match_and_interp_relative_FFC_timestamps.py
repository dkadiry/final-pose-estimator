import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Load the computed FFC relative poses (with RFC timestamps)
computed_ffc_poses = pd.read_csv("Updated_PF_Absolute_FFC_Pose_Quat_RFCTimestamps.csv")

# Load the ground truth FFC relative poses (with FFC timestamps)
ground_truth_ffc_poses = pd.read_csv('Data\Ground_truths\ground_truth_relative_FFC_poses.csv')

# Initialize a list to store final interpolated FFC relative poses with matched timestamps
final_ffc_poses_list = []

# Function to find the two closest timestamps and calculate the weight for interpolation
def find_closest_timestamps_and_weight(target_timestamp, reference_timestamps):
    differences = np.abs(reference_timestamps - target_timestamp)
    sorted_indices = np.argsort(differences)
    
    closest_index = sorted_indices[0]
    second_closest_index = sorted_indices[1] if len(sorted_indices) > 1 else closest_index
    
    if closest_index == second_closest_index:
        weight = 0.0  # No interpolation needed
    else:
        total_diff = differences[closest_index] + differences[second_closest_index]
        weight = differences[second_closest_index] / total_diff

    return closest_index, second_closest_index, weight

# Function to interpolate between two relative poses
def interpolate_poses(pose1, pose2, weight):
    # Linear interpolation for translation
    interpolated_translation = pose1[:3] * (1 - weight) + pose2[:3] * weight
    
    # Spherical linear interpolation (slerp) for rotation
    rot1 = R.from_quat(pose1[3:])
    rot2 = R.from_quat(pose2[3:])

    # Define key times for the rotations
    key_times = [0, 1]  # Start and end time for interpolation
    key_rots = R.from_quat([rot1.as_quat(), rot2.as_quat()])
    
    # Create Slerp object
    slerp = Slerp(key_times, key_rots)
    
    # Perform interpolation
    interpolated_rotation = slerp(weight).as_quat()
    
    return np.concatenate([interpolated_translation, interpolated_rotation])

# Iterate over each timestamp in the ground truth FFC relative poses
for i, row in ground_truth_ffc_poses.iterrows():
    target_timestamp = row['pose_timestamp']
    
    # Find the two closest timestamps in the computed FFC poses (with RFC timestamps)
    closest_index, second_closest_index, weight = find_closest_timestamps_and_weight(target_timestamp, computed_ffc_poses['pose_timestamp'])
    
    # Get the two closest poses
    pose1 = computed_ffc_poses.iloc[closest_index][['tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']].to_numpy()
    pose2 = computed_ffc_poses.iloc[second_closest_index][['tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']].to_numpy()
    
    # Interpolate to get the estimated FFC relative pose
    interpolated_pose = interpolate_poses(pose1, pose2, weight)
    
    # Append the interpolated relative pose with the ground truth FFC timestamp
    final_ffc_poses_list.append({
        'pose_timestamp': target_timestamp,
        'tx': interpolated_pose[0],
        'ty': interpolated_pose[1],
        'tz': interpolated_pose[2],
        'ox': interpolated_pose[3],
        'oy': interpolated_pose[4],
        'oz': interpolated_pose[5],
        'ow': interpolated_pose[6]
    })

# After the loop, concatenate the list of poses into a final DataFrame
final_ffc_poses = pd.DataFrame(final_ffc_poses_list)

# Save the final FFC relative poses with matched timestamps to a new CSV file
final_ffc_poses.to_csv("Pose_Results\particle_filtered_FFC_relative_pose_interp_timestamps_v2.csv", index=False)

print("Done matching timestamps and interpolating final FFC relative poses")
