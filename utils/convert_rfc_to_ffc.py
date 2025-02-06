import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


"""
This scripts converts the Absolute Pose Measurements of a Rear Facing Camera to the Absolute Pose Coordinates of the Front facing camera
where 

Constraints:
    The Poses are expressed as 3D position vectors and Quarternions
    Poses are stored in a CSV file
    The Origin for both cameras are their positions at initialization i.e. the same (0, 0, 0, 0, 0, 0, 1)
    Both cameras are mounted on the same platform simply facing opposite directions.

"""

# Load RFC pose data from CSV
rfc_poses = pd.read_csv('V6_Updated_PF_Absolute_RFC_Pose_Quat.csv')

# Function to eliminate small negative values close to 0
def clean_near_zero(value, threshold=1e-6):
    return 0 if abs(value) < threshold else value

def convert_rfc_to_ffc(row):
    #Capture timestamp
    ffc_timestamp = row['pose_timestamp']

    # Negate RFC translations to convert to FFC
    ffc_tx = clean_near_zero(-row['tx'])
    ffc_ty = clean_near_zero(-row['ty'])
    ffc_tz = clean_near_zero(-row['tz'])

    # Negate quaternion vector part and keep scalar part the same
    ffc_ox = clean_near_zero(-row['ox'])
    ffc_oy = clean_near_zero(-row['oy'])
    ffc_oz = clean_near_zero(-row['oz'])
    ffc_ow = row['ow']

    # Adjust translation with the new origin
    return pd.Series([ffc_timestamp, ffc_tx, ffc_ty, ffc_tz, ffc_ox, ffc_oy, ffc_oz, ffc_ow])

# Apply the transformation to each row
ffc_poses = rfc_poses.apply(lambda row: convert_rfc_to_ffc(row), axis=1)

# Rename columns to reflect FFC poses
ffc_poses.columns = ['pose_timestamp','tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow']

# Temporarily convert 'pose_timestamp' to integer (format with no decimal)
ffc_poses['pose_timestamp'] = ffc_poses['pose_timestamp'].apply(lambda x: f"{int(x):.0f}")

# Save FFC poses to a new CSV file
ffc_poses.to_csv('V6_Updated_PF_Absolute_FFC_Pose_Quat_RFCTimestamps.csv', index=False)

#Print Done to terminal
print("Done Conversion")