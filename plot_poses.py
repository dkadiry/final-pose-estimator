import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Example pose data: List of translations (tx, ty, tz) and quaternions (qx, qy, qz, qw)
#pose_data = [
    # [tx, ty, tz, qx, qy, qz, qw]
    #[0, 0, 0, 0, 0, 0, 1],
    #[1, 0, 0, 0, 0, 0.7071, 0.7071],
    #[2, 1, 0, 0, 0.7071, 0, 0.7071],
    #[3, 1, 1, 0, 0, 1, 0],
#]

pose_data_file = 'Pose_Results\Particle_filter_absolute_poses.csv'
pose_data = pd.read_csv(pose_data_file)

# Function to select a subset of poses
def select_pose_subset(data, start_idx, end_idx):
    """Select a subset of poses between start_idx and end_idx."""
    return data.iloc[start_idx:end_idx]

# User-defined subset selection
# Define the start and end index for the subset
start_idx = 0  # Example: Start index for the subset
end_idx = 482  # Example: End index for the subset (first 20 poses)

# Select the subset of data
subset_data = select_pose_subset(pose_data, start_idx, end_idx)

# Extract translations and quaternions from pose data
translations = subset_data[['tx', 'ty', 'tz']].values
quaternions = subset_data[['ox', 'oy', 'oz', 'ow']].values

# Apply transformation to switch to a left-handed system:
# Swap Y and Z to make Z upwards and Y into the screen (depth)
#translations[:, [1, 2]] = translations[:, [2, 1]]  # Swap Y and Z in translations

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot translations (positions) in 3D space
ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 'bo-', label='Path')

# Highlight the starting point in red
ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], color='red', s=100, label='Start', zorder=5)

# Highlight the ending point in green
ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], color='green', s=100, label='End', zorder=5)

"""
# Visualize orientation using quaternions (optional)
for i, quat in enumerate(quaternions):
    r = R.from_quat(quat)
    rot_matrix = r.as_matrix()

    # Swap Y and Z axis for the orientation vectors
    rot_matrix[:, [1, 2]] = rot_matrix[:, [2, 1]]  # Swap Y and Z in orientation

    # Create an arrow for the orientation based on the forward direction
    start_point = translations[i]
    end_point = start_point + rot_matrix[:, 0] * 0.5  # Scale for visibility

    ax.quiver(
        start_point[0], start_point[1], start_point[2],  # Starting point
        end_point[0] - start_point[0],  # X-direction of the arrow
        end_point[1] - start_point[1],  # Y-direction
        end_point[2] - start_point[2],  # Z-direction
        color='y', length=0.5, normalize=True, label='Orientation' if i == 0 else ""
    )
"""


# Labels and plot adjustments
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Pose Visualization from CSV')
ax.legend()

# Show plot
plt.show()