import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd


def rename_files_to_timestamps(folder_path):
    """
    Rename all .jpg and .txt files in the given folder to just their timestamps.
    
    The function assumes the filenames contain a timestamp after 'zed_image_left_' 
    and will rename the files to just the timestamp.
    
    Parameters:
    folder_path (str): The path to the folder containing the files to be renamed.
    """

    index = 1

    # Loop through each file and rename
    for filename in os.listdir(folder_path):
        
        if filename.endswith(".txt") or filename.endswith(".jpg") or filename.endswith(".npy"):
            # Extract timestamp
            timestamp = filename.split("zed_image_left_")[-1].split("_")[0]
            #timestamp = filename.split("_")[0]

            # Define newname
            file_ext = os.path.splitext(filename)[1]
            new_name = timestamp + file_ext
            
            # Define old and new path for rename function
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)

            # Rename File
            os.rename(old_file_path, new_file_path)
            print(f"Renamed file number {index}: {filename} -> {new_name}")
            index += 1

    print("All files renamed")


def find_mismatch(folder_1, folder_2):

    # Get timestamps
    timestamps_1 = {os.path.splitext(filename)[0] for filename in os.listdir(folder_1)}
    timestamps_2 = {os.path.splitext(filename)[0] for filename in os.listdir(folder_2)}

    missing_in_one = timestamps_2 - timestamps_1
    missing_in_two = timestamps_1 - timestamps_2
    
    return list(missing_in_one), list(missing_in_two)


def load_image_pair_and_labels(image_folder, label_folder, index):
    # List all image and label files
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    
    # Ensure index is within bounds
    if index >= len(image_files) - 1:
        return None, None, None, None, index
    
    # Load the current image and the next one
    image1_path = os.path.join(image_folder, image_files[index])
    image2_path = os.path.join(image_folder, image_files[index + 1])
    
    label1_path = os.path.join(label_folder, label_files[index])
    label2_path = os.path.join(label_folder, label_files[index + 1])
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    with open(label1_path, 'r') as file:
        labels1 = file.readlines()
    with open(label2_path, 'r') as file:
        labels2 = file.readlines()

    filename1 = os.path.basename(image1_path)
    filename2 = os.path.basename(image2_path)

    timestamp1 = re.findall(r'\d+', filename1)[0]
    timestamp2 = re.findall(r'\d+', filename2)[0]
    
    # Return images, labels, and the next index
    return image1, image2, labels1, labels2, index + 1, timestamp1, timestamp2, image1_path, image2_path


def undistort_images (image1, image2):
    # Define Camera Instrinsics (Obtained from Zed Camera calibration file)
    K = np.array([[700.425, 0, 642.845],
                           [0, 700.425, 395.8775],
                           [0,0,1]])
    

    # Distortion Coefficients (obtained from Zed Camera Calibration file)
    dist_coeffs = np.array([-0.170999, 0.0238468, 0, 0, 0])


    # Undistort both images
    Undistorted_im1 = cv2.undistort(image1, K, dist_coeffs)
    Undistorted_im2 = cv2.undistort(image2, K, dist_coeffs)

    return Undistorted_im1, Undistorted_im2

if __name__ == "__main__":

    """
    Use case for File renaming
    #Set the path to the folder containing the text/jpg files
    path = "Data\Bounding_box_labels"
    rename_files_to_timestamps(path)

    
    Use Case for matching pose data to image data

    images_folder = "Data\RFC Pose Estimation Images"
    labels_folder = "Data\Bounding_box_labels"

    missing_in_images, missing_in_labels = find_mismatch(images_folder, labels_folder)

    if missing_in_images:
        print(f"Timestamps in labels but not in images: {missing_in_images}")

    if missing_in_labels:
        print(f"Timestamps in images but not in labels: {missing_in_labels}")

    if not missing_in_labels and not missing_in_images:
        print("All timestamps match between the two folders.")


    """
    
    """

    Example of me retrieving rotation in all it's values    
    print(f"Relative Rotation: {rotation_mat}\n  Relative translation (Unscaled): {translation_vec}")

    
      

    quart_obj = R.from_quat(rotation_quaternion)
    euler_angles = quart_obj.as_euler('xyz', degrees=True)


    reconstruct_r = R.from_euler('xyz', euler_angles, degrees=True)
    

    reconstruct_mat = reconstruct_r.as_matrix()
    
    roll, pitch, yaw = euler_angles
    print(f'Quarternion: {rotation_quaternion}')
    print(f"Roll: {roll:.6f} degrees")
    print(f"Pitch: {pitch:.6f} degrees")
    print(f"Yaw: {yaw:.6f} degrees")

    print(f"Reconstructed Rotation Matrix from Retreived Eulers: {reconstruct_mat}")

    #Example Usage for Loading two image frames and undistortion

    image_folder = "Data\RFC Pose Estimation Images"
    label_folder = "Data\Bounding_box_labels"
    index = 0

    image1, image2, _, label2, index, _, _ = load_image_pair_and_labels(image_folder, label_folder, 0)

    for label in label2:
        print(label)
    

    new_im1, new_im2 = undistort_images(image1, image2)

    # Compare original and undistorted visually
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Undistorted image")
    plt.imshow(cv2.cvtColor(new_im1, cv2.COLOR_BGR2RGB))

    plt.show()
    
  


    
    
    """

    # Load the ground truth and estimated data
    ground_truth_df = pd.read_csv("Data/Ground_truths/euler_relative_poses_ground_truth.csv")  # Replace with your ground truth file path
    estimated_df = pd.read_csv("Pose_Results/euler_relative_poses_approach5_5.csv")  # Replace with your estimated data file path

    # Extract timestamps
    ground_truth_timestamps = set(ground_truth_df['pose_timestamp'])
    estimated_timestamps = set(estimated_df['pose_timestamp'])

    # Find missing timestamps
    missing_in_estimated = ground_truth_timestamps - estimated_timestamps
    missing_in_ground_truth = estimated_timestamps - ground_truth_timestamps

    # Display the missing timestamps
    print("Timestamps in ground truth but missing in estimated data:", sorted(missing_in_estimated))
    print("Timestamps in estimated data but missing in ground truth:", sorted(missing_in_ground_truth))

    
