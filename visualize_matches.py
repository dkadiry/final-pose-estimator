#import utils.tools as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import csv
import os
from statistics import mean
import math
from PIL import Image, ImageDraw
import re

# Set random seeds for reproducibility
np.random.seed(42)          # Seed for NumPy random operations
cv2.setRNGSeed(42)          # Seed for OpenCV random operations

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
    return image1, image2, labels1, labels2, index + 1, timestamp1, timestamp2


def detect_and_compute_points(image1, image2):
    cv2.setRNGSeed(42)
    # Initialize sift detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    #print(f"Number of keypoints in image1: {len(keypoints_1)}, image2: {len(keypoints_2)}")

    if descriptors_1 is not None and descriptors_2 is not None:
        # Create a FLANN based matcher and match descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)
        search_params = dict(checks=200)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Apply the ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        #print(f"Number of good matches: {len(good_matches)}")

        # Visualize the matches
        #plt.figure()
        #img_matches1 = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #plt.title('Matches Before RANSAC')
        #plt.imshow(img_matches1), plt.show()
        
        return keypoints_1, keypoints_2, good_matches
    return [], [], [] # If descriptors are none, return empty lists

def compute_essential_matrix(keypoints1, keypoints2, matches, cam_matrix):
    cv2.setRNGSeed(42)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros_like(points1)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find Essential Matrix
    E, mask = cv2.findEssentialMat(points1, points2, cam_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Filter points using the RANSAC mask
    inlier_pts1 = points1[mask.ravel() == 1]
    inlier_pts2 = points2[mask.ravel() == 1]
    #print(f"Number of inlier points in image 1: {len(inlier_pts1)}, Number of inlier points in image 2: {len(inlier_pts2)}")
    # Recover relative pose
    _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, cam_matrix)

    return R, t

def convert_mat_to_quat(rotation_mat):
    r = R.from_matrix(rotation_mat)
    rotation_quaternion = r.as_quat()
   
    return rotation_quaternion

def convert_quat_to_mat(rotation_quat):
    quart_obj = R.from_quat(rotation_quat)
    reconstructed_mat = quart_obj.as_matrix()

    return reconstructed_mat

def convert_quaternion_to_euler(rotation_quat):
       
    # Convert quaternion to Euler angles
    r = R.from_quat(rotation_quat)
    euler_angles = r.as_euler('xyz', degrees=False)  # Use radians
    
    # Return the Euler angles as new columns
    return euler_angles[0], euler_angles[1], euler_angles[2]

def main():
    
    image_folder = "Data\RFC Pose Estimation Images"
    label_folder = "Data\Bounding_box_labels"
    index = 148
    is_scaled = 0
    
    # Data Storage for Pose data
    pose_data_file = os.path.join("Pose_Results", "ignore_RFC_relative_poses_approach5_5.csv")
    with open(pose_data_file, 'w') as f:
        f.write("pose_timestamp,tx,ty,tz,ox,oy,oz,ow,is_scaled,object_used\n")  # CSV header
    
    # Define Camera Instrinsics (Obtained from Zed Camera calibration file)
    K = np.array([[700.425, 0, 642.845],
                           [0, 700.425, 395.8775],
                           [0,0,1]])
    
    num_runs = 3 # Set number of runs (Total number needed for full run is 1639)

    for i in range(num_runs):
        # Load in the next pair of images and corresponding labels
        print(f"Index is for run {i} is {index}")
        im1, im2, label1, label2, index, ts1, ts2 = load_image_pair_and_labels(image_folder, label_folder, index) 
        print (f"Image 1 is {ts1} and Image 2 is {ts2}")
        
        # Compute SIFT points and return matches using FLANN K-Nearest neighbour
        kp1, kp2, matches = detect_and_compute_points(im1, im2) 
        #print(f"Good matches for index {index-1}: {len(matches)}")

        # Compute the rotation matrix and translation vector from the essential matrix using RANSAC
        rotation_mat, translation_vec = compute_essential_matrix(kp1, kp2, matches, K)

        print(rotation_mat)

        rotation_quat = convert_mat_to_quat(rotation_mat)
        pitch, yaw, roll = convert_quaternion_to_euler(rotation_quat)
        print(f"Pitch = {pitch}, Yaw = {yaw}, roll = {roll}")
        
        
    print("Done Visualization")
       
    
if __name__ == '__main__':
    main()