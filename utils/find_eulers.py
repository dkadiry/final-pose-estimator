import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the CSV file containing poses (with quaternions/eulers)
relative_poses = pd.read_csv('V6_Updated_PF_Absolute_FFC_Pose_Quat.csv')

def convert_quaternion_to_euler(row):
    # Extract quaternion components from the row
    qx, qy, qz, qw = row['ox'], row['oy'], row['oz'], row['ow']
    
    # Convert quaternion to Euler angles
    r = R.from_quat([qx, qy, qz, qw])
    euler_angles = r.as_euler('xyz', degrees=False)  # Use radians
    
    # Return the Euler angles as new columns
    return pd.Series([euler_angles[0], euler_angles[1], euler_angles[2]])

def convert_euler_to_quarternion(row):
    # Extract Euler Components from the row
    pitch, yaw, roll = row['pitch'], row['yaw'], row['roll']

    # Create a rotation object using the Euler angles in 'xyz' order
    rotation_euler = R.from_euler('xyz', [pitch, yaw, roll], degrees=False)

    # Convert to quaternion
    quarternion = rotation_euler.as_quat()

    #Ensure the quarternions are normalized
    quarternion /= np.linalg.norm(quarternion)

    return pd.Series([quarternion[0], quarternion[1], quarternion[2], quarternion[3]])

def create_euler_csv(poses):
    # Apply the quaternion to Euler conversion to each row
    euler_poses = poses.apply(convert_quaternion_to_euler, axis=1)

    # Rename the columns to reflect the Euler angles (roll, pitch, yaw)
    euler_poses.columns = ['pitch', 'yaw', 'roll']

    # Combine the Euler angles with the original dataframe
    #new_relative_poses = pd.concat([poses[['pose_timestamp','tx', 'ty', 'tz']], euler_poses, poses[['is_scaled']]], axis=1)
    new_relative_poses = pd.concat([poses[['pose_timestamp','tx', 'ty', 'tz']], euler_poses], axis=1)

    # Save the new relative poses to a new CSV file
    new_relative_poses.to_csv('V6_Updated_PF_Absolute_FFC_Pose_Euler.csv', index=False)

    # Print Done to terminal
    print("Done Conversion to Eulers")

def create_quat_csv(poses):
    #Apply the Euler to quartenion conversion to each row
    quat_poses = poses.apply(convert_euler_to_quarternion, axis=1)

    #Rename the columns to reflect the quaternion components
    quat_poses.columns = ['ox', 'oy', 'oz', 'ow']

    # Combine the quat pose with the original dataframe
    #new_relative_poses = pd.concat([poses[['pose_timestamp','tx', 'ty', 'tz']], quat_poses, poses[['is_scaled', 'object_used']]], axis=1)
    new_relative_poses = pd.concat([poses[['pose_timestamp','tx', 'ty', 'tz']], quat_poses], axis=1)

    #Save the new relative poses to a new CSV file
    new_relative_poses.to_csv("V6_Updated_PF_Absolute_RFC_Pose_Quat.csv", index=False)

    print("Done Converting to Quarternions")


if __name__ == "__main__":
    create_euler_csv(relative_poses)
    #create_quat_csv(relative_poses)