import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Load the converted RFC poses from CSV (these have FFC timestamps)
converted_rfc_poses = pd.read_csv("Data\Ground_truths\converted_groundtruth_RFC_poses.csv")

# Load the ground truth RFC poses from CSV (these have ground truth RFC timestamps)
ground_truth_rfc_poses = pd.read_csv('Data\Copy of RFC Pose Data.csv')

# Initialize a new DataFrame for the final FFC poses with matched timestamps
#final_ffc_poses = pd.DataFrame(columns=['timestamp', 'tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow'])
final_rfc_poses_list = []

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

# Function to interpolate between two poses
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


# Copy the first pose directly from the converted RFC poses
first_pose = converted_rfc_poses.iloc[0]
first_pose_timestamp = ground_truth_rfc_poses.iloc[0]['zed_timestamp']
final_rfc_poses_list.append({
    'pose_timestamp': first_pose_timestamp,
    'tx': first_pose['tx'],
    'ty': first_pose['ty'],
    'tz': first_pose['tz'],
    'ox': first_pose['ox'],
    'oy': first_pose['oy'],
    'oz': first_pose['oz'],
    'ow': first_pose['ow']
})


# Iterate over each timestamp in the ground truth RFC poses starting from pose 2
for i in range(1, 1640):  # Start from the second pose
    if i >= 1640:  # We are only interested in the first 1640 ground truth RFC timestamps
        break
    
    target_timestamp = ground_truth_rfc_poses.iloc[i]['zed_timestamp']
    
    # Find the two closest timestamps in the converted RFC poses (with RFC timestamps)
    closest_index, second_closest_index, weight = find_closest_timestamps_and_weight(target_timestamp, converted_rfc_poses['pose_timestamp'])
    
    # Get the two closest poses
    pose1 = converted_rfc_poses.iloc[closest_index][['tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']].to_numpy()
    pose2 = converted_rfc_poses.iloc[second_closest_index][['tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']].to_numpy()
    
    # Interpolate to get the estimated RFC pose
    interpolated_pose = interpolate_poses(pose1, pose2, weight)
    
    # Append the interpolated pose to the final DataFrame with the ground truth RFC timestamp
    final_rfc_poses_list.append({
        'pose_timestamp': ground_truth_rfc_poses.iloc[i]['zed_timestamp'],
        'tx': interpolated_pose[0],
        'ty': interpolated_pose[1],
        'tz': interpolated_pose[2],
        'ox': interpolated_pose[3],
        'oy': interpolated_pose[4],
        'oz': interpolated_pose[5],
        'ow': interpolated_pose[6]
    })

# After the loop, concatenate the list of poses into a final DataFrame
final_rfc_poses = pd.concat([pd.DataFrame([pose]) for pose in final_rfc_poses_list], ignore_index=True)

# Temporarily convert 'pose_timestamp' to integer (format with no decimal)
final_rfc_poses['pose_timestamp'] = final_rfc_poses['pose_timestamp'].apply(lambda x: f"{int(x):.0f}")

# Save the final RFC poses with matched timestamps to a new CSV file
final_rfc_poses.to_csv("Data/Ground_truths/interpolated_ground_truth_RFC_poses.csv", index=False)

print("Done matching timestamps and interpolating ground truth RFC pose")
