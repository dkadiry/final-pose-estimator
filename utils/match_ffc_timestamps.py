import pandas as pd
import numpy as np

# Load the converted FFC poses from CSV (these have RFC timestamps)
converted_ffc_poses = pd.read_csv('Pose_Results\converted_alt_raw_ffc_poses.csv')

# Load the ground truth FFC poses from CSV (these have ground truth FFC timestamps)
ground_truth_ffc_poses = pd.read_csv('Data\Copy of FFC Pose Data.csv')

# Initialize a new DataFrame for the final FFC poses with matched timestamps
final_ffc_poses = pd.DataFrame(columns=['timestamp', 'tx', 'ty', 'tz', 'ox', 'oy', 'oz', 'ow'])

# Function to find the closest timestamp
def find_closest_timestamp(target_timestamp, reference_timestamps):
    differences = np.abs(reference_timestamps - target_timestamp)
    closest_index = differences.argmin()
    return closest_index

# Iterate over each timestamp in the ground truth FFC poses
for i, row in ground_truth_ffc_poses.iterrows():
    if i >= 1640:  # We are only interested in the first 1640 ground truth FFC timestamps
        break
    
    target_timestamp = row['zed_timestamp']
    
    # Find the closest timestamp in the converted FFC poses (with RFC timestamps)
    closest_index = find_closest_timestamp(target_timestamp, converted_ffc_poses['pose_timestamp'])
    
    # Use the closest pose from the converted FFC poses for the final FFC pose
    closest_pose = converted_ffc_poses.iloc[closest_index]
    
    # Append the pose to the final DataFrame with the ground truth FFC timestamp
    final_ffc_poses = final_ffc_poses.append({
        'timestamp': row['zed_timestamp'],
        'tx': closest_pose['tx'],
        'ty': closest_pose['ty'],
        'tz': closest_pose['tz'],
        'ox': closest_pose['ox'],
        'oy': closest_pose['oy'],
        'oz': closest_pose['oz'],
        'ow': closest_pose['ow']
    }, ignore_index=True)

# Save the final FFC poses with matched timestamps to a new CSV file
final_ffc_poses.to_csv('final_alt_raw_ffc_poses.csv', index=False)
print("Done")