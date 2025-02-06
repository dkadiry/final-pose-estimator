import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the absolute pose data from CSV
absolute_poses = pd.read_csv('V6_Updated_PF_Absolute_FFC_Pose_Quat.csv')

def compute_relative_pose(df):
    relative_poses = []

    # Loop through rows, starting from the second pose (index 1) to calculate relative pose
    for i in range(1, len(df)):
        # Extract current and previous translations
        prev_translation = np.array([df.iloc[i-1]['tx'], df.iloc[i-1]['ty'], df.iloc[i-1]['tz']])
        curr_translation = np.array([df.iloc[i]['tx'], df.iloc[i]['ty'], df.iloc[i]['tz']])

        # Extract current and previous quaternions
        prev_quaternion = np.array([df.iloc[i-1]['ox'], df.iloc[i-1]['oy'], df.iloc[i-1]['oz'], df.iloc[i-1]['ow']])
        curr_quaternion = np.array([df.iloc[i]['ox'], df.iloc[i]['oy'], df.iloc[i]['oz'], df.iloc[i]['ow']])

        # Ensure Normalize quaternions
        prev_quaternion /= np.linalg.norm(prev_quaternion)
        curr_quaternion /= np.linalg.norm(curr_quaternion)
        
        # Convert quaternions to Rotation objects
        prev_rot = R.from_quat(prev_quaternion)
        curr_rot = R.from_quat(curr_quaternion)

        # Compute relative rotation: Q_relative = Q_prev^(-1) * Q_curr
        relative_rot = prev_rot.inv() * curr_rot
        relative_quat = relative_rot.as_quat()  # Get the quaternion representation

        # Compute relative translation
        relative_translation_global = curr_translation - prev_translation

        # Rotate the global translation into the previous local frame
        relative_translation = prev_rot.inv().apply(relative_translation_global)

        # Store the relative pose (timestamp, translation, rotation)
        relative_poses.append([
            df.iloc[i]['pose_timestamp'],  # Current timestamp
            relative_translation[0], relative_translation[1], relative_translation[2],  # Relative translation
            relative_quat[0], relative_quat[1], relative_quat[2], relative_quat[3]  # Relative quaternion
        ])
    
    # Convert list of relative poses to a DataFrame
    relative_poses_df = pd.DataFrame(relative_poses, columns=['pose_timestamp', 'tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow'])
    
    return relative_poses_df

# Compute relative poses
relative_poses = compute_relative_pose(absolute_poses)

relative_poses['pose_timestamp'] = relative_poses['pose_timestamp'].apply(lambda x: f"{int(x):.0f}")

# Save the relative poses to a new CSV file
relative_poses.to_csv('V6_Updated_PF_Relative_FFC_Pose_Quat.csv', index=False)

# Print Done to terminal
print("Done Conversion")