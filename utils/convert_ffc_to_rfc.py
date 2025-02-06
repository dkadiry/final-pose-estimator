import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load RFC pose data from CSV
ffc_poses = pd.read_csv('Data\Ground_truths\copy_Corrected_ground_truth_FFC_pose.csv')

# Define the FFC starting quaternion and RFC starting quaternion (origin offset)
ffc_starting_quaternion = ffc_poses.iloc[0][['ox', 'oy', 'oz', 'ow']].values

# Find the inverse of the FFC starting quaternion
inverse_ffc_quaternion = R.from_quat(ffc_starting_quaternion).inv().as_quat()

# Function to eliminate small negative values close to 0
def clean_near_zero(value, threshold=1e-6):
    return 0 if abs(value) < threshold else value

def convert_ffc_to_rfc(row):
    #Capture timestamp
    rfc_timestamp = row['pose_timestamp']
   
    # Negate FFC translations to convert to RFC
    rfc_tx = clean_near_zero(-row['tx'])
    rfc_ty = clean_near_zero(-row['ty'])
    rfc_tz = clean_near_zero(-row['tz'])

    #Flip the signs of the quaternion components (x, y, z)
    flipped_quat = [-row['ox'], -row['oy'], -row['oz'], row['ow']]

    #Clean up any small values near 0 (positive or negative)
    cleaned_quat = [clean_near_zero(q) for q in flipped_quat]  

     # Adjust translation with the new origin translation
    adjusted_tx = rfc_tx 
    adjusted_ty = rfc_ty 
    adjusted_tz = rfc_tz

    return pd.Series([rfc_timestamp, adjusted_tx, adjusted_ty, adjusted_tz, cleaned_quat[0], cleaned_quat[1], cleaned_quat[2], cleaned_quat[3]])

# Apply the transformation to each row
rfc_poses = ffc_poses.apply(lambda row: convert_ffc_to_rfc(row), axis=1)

# Rename columns to reflect FFC poses
rfc_poses.columns = ['pose_timestamp','tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']

# Temporarily convert 'pose_timestamp' to integer (format with no decimal)
rfc_poses['pose_timestamp'] = rfc_poses['pose_timestamp'].apply(lambda x: f"{int(x):.0f}")

# Save FFC poses to a new CSV file
rfc_poses.to_csv('Data/Ground_truths/converted_groundtruth_RFC_poses.csv', index=False)

#Print Done to terminal
print("Done Conversion")