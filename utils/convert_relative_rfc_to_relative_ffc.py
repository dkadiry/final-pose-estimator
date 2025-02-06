import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load RFC relative pose data from CSV
rfc_poses = pd.read_csv('Pose_Results\Final_Approach_results\Particle Filter Estimates\RFC Estimates\PF_relative_poses_with_quat.csv')

# Define the new origin quaternion for FFC
new_origin_quaternion = np.array([-0.027, 0, 0.017, 0.999])  # New FFC origin rotation quaternion

# Function to convert each relative RFC pose to a relative FFC pose
def convert_relative_rfc_to_relative_ffc(row, new_origin_quat):
    # Capture timestamp
    ffc_timestamp = row['pose_timestamp']

    # Convert RFC relative translation to FFC by negating values
    ffc_tx = -row['tx']
    ffc_ty = -row['ty']
    ffc_tz = -row['tz']

    # Convert RFC relative rotation to FFC by negating the vector part of the quaternion
    ffc_ox = -row['ox']
    ffc_oy = -row['oy']
    ffc_oz = -row['oz']
    ffc_ow = row['ow']

    # Combine with the new FFC origin rotation
    rfc_rot = R.from_quat([ffc_ox, ffc_oy, ffc_oz, ffc_ow])
    origin_rot = R.from_quat(new_origin_quat)
    combined_rot = origin_rot * rfc_rot  # Apply FFC origin rotation
    combined_quat = combined_rot.as_quat()

    # Return relative FFC pose with transformed rotation and translation
    return pd.Series([ffc_timestamp, ffc_tx, ffc_ty, ffc_tz, combined_quat[0], combined_quat[1], combined_quat[2], combined_quat[3]])

# Apply the transformation to each row to get relative FFC poses
ffc_relative_poses = rfc_poses.apply(lambda row: convert_relative_rfc_to_relative_ffc(row, new_origin_quaternion), axis=1)

# Rename columns to reflect FFC relative poses
ffc_relative_poses.columns = ['pose_timestamp', 'tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']

# Save FFC relative poses to a new CSV file
ffc_relative_poses.to_csv('Pose_Results\particle_filtered_FFC_relative_pose_RFCtimestamps_v2.csv', index=False)

# Print Done to terminal
print("Done Conversion")
