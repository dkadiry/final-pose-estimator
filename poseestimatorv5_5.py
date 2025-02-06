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


def find_camera_translation_vector(image1, image2, labels1, labels2, camera_intrinsic, rotation_matrix):
    # Parse labels to find the "Chair" class
    chair_class_id = 3  
    track_class_id = 9
    box_id = 2
    black_bag_id = 0
    log_id = 4
    monitor_id = 5
    recycle_box_id = 6
    red_bag_id = 8
    bboxes1 = []
    bboxes2 = []
   

    # Parse through each line in the bounding box label text file and convert their coordinates to pixels, then store them in a list of bounding boxes
    for label in labels1:
        class_id = int(label.split()[0])
        bbox = get_bounding_box_pixels(label, image1.shape[0], image1.shape[1])
        bboxes1.append((class_id, bbox))
    #print(f"Bounding boxes for image1: {bboxes1}")

    # Parse through each line in the second bounding box label text file and convert their coordinates to pixels, then store them in a list of bounding boxes
    for label in labels2:
        class_id = int(label.split()[0])
        bbox = get_bounding_box_pixels(label, image2.shape[0], image2.shape[1])
        bboxes2.append((class_id, bbox))
    #print(f"Bounding boxes for image2: {bboxes2}")
    

    # Check each bounding box list to see if they have a chair detected first
    chairs1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == chair_class_id]
    chairs2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == chair_class_id]

    # If there is at least one chair in both images
    if chairs1 and chairs2:

        matched_chairs = match_chairs(chairs1, chairs2) # Find a pair of matching chairs (Need to do this in case there are 2 chairs in at least one image)
        
        #print(matched_chairs)
        if matched_chairs:
            chair_height_meters = 0.8          
            chair1, chair2 = choose_largest_chair(matched_chairs) # Choose the largest pair of matching chairs (Need to do this if there is more than one chair in both images)
            trans_vector = compute_vector(camera_intrinsic, rotation_matrix, chair_height_meters, chair1, chair2)
            object_name = "chair"
            print(f"Object used is {object_name}")
            return trans_vector, object_name
        else:
            print("No matching chairs found between the two images.")

    # Check for other objects if no chairs found
    print("No chair found in one or both images, looking for alternative objects")
    filtered_bboxes1 = filter_classes(bboxes1)
    filtered_bboxes2 = filter_classes(bboxes2)

    # If alternate ojects found in image one and two
    if filtered_bboxes1 and filtered_bboxes2:
        common_classes = find_common_object(filtered_bboxes1, filtered_bboxes2)

        # If the same alternate objects are found in image one and two
        if common_classes:
            alt_object1, alt_object2 = find_alternative_object(filtered_bboxes1, filtered_bboxes2, common_classes) # Select the largest alternate object pair
            if alt_object1 and alt_object2:
                alt_bbox1 = alt_object1[1]
                alt_bbox2 = alt_object2[1]
                
                height_dict = {
                    black_bag_id: 0.28,
                    box_id: 0.31,
                    log_id: 0.28,
                    monitor_id: 0.40,
                    red_bag_id: 0.28
                }
                object_height_meters = height_dict.get(alt_object1[0], None)
                object_name = {
                    black_bag_id: "Black Bag",
                    box_id: "Box",
                    log_id: "Log",
                    monitor_id: "Monitor",
                    red_bag_id: "Red Bag"
                }.get(alt_object1[0], None)

                if object_height_meters:
                    trans_vector = compute_vector(camera_intrinsic, rotation_matrix, object_height_meters, alt_bbox1, alt_bbox2)
                    print(f"Object is {object_name}")
                    return trans_vector, object_name

    # Check for recycle bins if no other objects found
    print("No bags, monitor, box, or log found. Checking for recycle bins...")
    recycle_bins1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == recycle_box_id]
    recycle_bins2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == recycle_box_id]

    if recycle_bins1 and recycle_bins2:
        matched_bins = match_bins(recycle_bins1, recycle_bins2) # Make sure the recycle bins are the same bin
        if matched_bins:
            recycle_box1, recycle_box2 = matched_bins
            bin_height_meters = 0.33  # Real-world height of recycle bin in meters
            trans_vector = compute_vector(camera_intrinsic, rotation_matrix, bin_height_meters, recycle_box1, recycle_box2)
            object_name = "Recycle bin"
            print(f"Object is {object_name}")
            return trans_vector, object_name

    # Check for tracks if no recycle bins found
    print("No recycle bin found or bins do not match. Checking for tracks...")
    tracks1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == track_class_id]
    tracks2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == track_class_id]

    if len(tracks1) >= 2 and len(tracks2) >= 2:
        if are_tracks_side_by_side(tracks1) and are_tracks_side_by_side(tracks2):
            distance_im1, centre_im1 = calculate_track_distance_and_centre(tracks1)
            distance_im2, centre_im2 = calculate_track_distance_and_centre(tracks2)
            real_world_width = 0.67  # Real-world distance between tracks in meters
            
            # Define Variables
            cam_matrix_inv = np.linalg.inv(camera_intrinsic) # Inverse of Camera Intrinsic Matrix
            f_length = 700.425 # Obtained from Camera Calibration file. Same in X and Y

            # Define 2D homogenous pixel coordinates vector for track center in both images
            hom_centre_im1 = np.array([*centre_im1, 1])
            hom_centre_im2 = np.array([*centre_im2, 1])

            # Find distances to track center in meters for both images 
            dist_centre1 = (f_length * real_world_width) / distance_im1
            dist_centre2 = (f_length * real_world_width) / distance_im2

            # Compute 3D coordinates of the track centre in image 1
            track_X1, track_Y1, track_Z1 = dist_centre1 * np.dot(cam_matrix_inv, hom_centre_im1)
            
            track_P1 = np.array([track_X1, track_Y1, track_Z1]) # Track center 3D position with camera in image 1 set as origin

            # Find camera translation vector
            rot_P1 = np.dot(rotation_matrix, track_P1) # Rotation matrix @ P1 vector => 3 x 1 vector 

            cam_KR_P1 = np.dot(camera_intrinsic, rot_P1) # K * R * P1 => 3 x 1 vector

            z2_pixel_coords = dist_centre2 * hom_centre_im2 # Scaled homogenous pixel coordinates in image 2 => 3 x 1 vector

            difference = z2_pixel_coords - cam_KR_P1 

            trans_vector = np.dot(cam_matrix_inv, difference) # => 3x1 translation vector = K^-1 * (Z2 * hom_pixel_coords2 - K @ R @ P1)

            object_name = "Track"
            print(f"Object used is {object_name}")
            return trans_vector, object_name
        print("Tracks found but they are not side by side")

    # Final else to catch all other cases
    print("No objects found for scale factor calculation")
    return None

def compute_vector(cam_intrinsic, rot_mat, height_meters, bbox1, bbox2):
    # Define Variables
    K_inv = np.linalg.inv(cam_intrinsic) # Inverse of Camera Intrinsic Matrix
    cam_K = cam_intrinsic
    focal_length = 700.425 # Obtained from Camera Calibration file. Same in X and Y

    # Find Object Height in pixels from both images
    object_pixel_height_1 = bbox1[3]
    object_pixel_height_2 = bbox2[3]

    #print(f"Height in 1: {object_pixel_height_1}")
    #print(f"Height in 1: {object_pixel_height_2}")
    
    # Define 2D homogenous pixel coordinates vector for object center in both images, where bbox (x_center, y_center, width, height)
    hom_pixel_coord1 = np.array([bbox1[0], bbox1[1], 1])
    hom_pixel_coord2 = np.array([bbox2[0], bbox2[1], 1])
    #print(hom_pixel_coord1)
    #print(hom_pixel_coord2)

    # Find distances to object in meters for both images 
    dist_object1 = (focal_length * height_meters) / object_pixel_height_1
    dist_object2 = (focal_length * height_meters) / object_pixel_height_2

    #print("Distance in 1",dist_object1)
    #print ("Distance in 2",dist_object2)
   
    # Compute 3D coordinates of the object in image 1
    obj1_X, obj1_Y, obj1_Z = dist_object1 * np.dot(K_inv, hom_pixel_coord1)

    #obj2_X, obj2_Y, obj2_Z = dist_object2 * np.dot(K_inv, hom_pixel_coord2)
    
    object_P1 = np.array([obj1_X, obj1_Y, obj1_Z]) # Object 3D position with camera in image 1 set as origin
    #object_P2 = np.array([obj2_X, obj2_Y, obj2_Z ])
    #print(object_P1)
    #print(rot_mat)
    # Find camera translation vector
    rot_P1 = np.dot(rot_mat, object_P1) # Rotation matrix @ P1 vector => 3 x 1 vector 

    cam_KR_P1 = np.dot(cam_K, rot_P1) # K * R * P1 => 3 x 1 vector

    z2_pixel_coords = dist_object2 * hom_pixel_coord2 # Scaled homogenous pixel coordinates in image 2 => 3 x 1 vector

    difference = z2_pixel_coords - cam_KR_P1 

    translation_vec = np.dot(K_inv, difference) # => 3x1 translation vector = K^-1 * (Z2 * hom_pixel_coords2 - K @ R @ P1)

    #op2 = np.dot(rot_mat, object_P1) + translation_vec

    #print(f"Object in cam 2 using cam2 as origin {object_P2}")
    #print(f"Object in cam 2 from our calculation {op2}")

    return translation_vec

    
def get_bounding_box_pixels(label, im_height, im_width):
    id, x_center, y_center, width, height = map(float, label.split())
    x_center_pixel = int(x_center * im_width)
    y_center_pixel = int(y_center * im_height)
    width_pixel = int(width * im_width)
    height_pixel = int(height * im_height)

    #print (f"Bounding Box in YOLO {id, x_center, y_center, width, height}, BBox in pixel {x_center_pixel, y_center_pixel, width_pixel, height_pixel}")
    
    return x_center_pixel, y_center_pixel, width_pixel, height_pixel

def filter_classes(data):
    return [(class_id, bbox) for class_id, bbox in data if class_id not in [1, 3, 7, 9]]

def find_common_object(bboxes1, bboxes2):
    class_ids1 = {class_id for class_id, _ in bboxes1}
    class_ids2 = {class_id for class_id, _ in bboxes2}

    common_objects = list(class_ids1.intersection(class_ids2))
    return common_objects

def find_alternative_object(bboxes1, bboxes2, common_objects):
    max_area = float('-inf')
    chosen_object1 = None

    common_list1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id in common_objects]
        
    for class1, bbox1 in common_list1:
        area1 = compute_area(bbox1)
        if area1 > max_area:
            max_area = area1
            chosen_object1 = (class1, bbox1)

    chosen_object2 = None
    if chosen_object1:
        for classid2, bbox2 in bboxes2:
            if classid2 == chosen_object1[0]:
                chosen_object2 = (classid2, bbox2)
                break

    return chosen_object1, chosen_object2


def match_chairs(chairs1, chairs2):
    matched_chairs = []
    #print(f"Number of chairs in image 1: {len(chairs1)}")
    #print(f"Number of chairs in image 2: {len(chairs2)}")
    #print(f"Chairs in image 1: {chairs1}")
    #print(f"Chairs in image 2: {chairs2}")

    if len(chairs1) > 1 or len(chairs2) > 1:
        if len(chairs1) < len(chairs2):
            outer_list = chairs1
            inner_list = chairs2
        elif len(chairs2) < len(chairs1):
            outer_list = chairs2
            inner_list = chairs1
        else:
            outer_list = chairs1
            inner_list = chairs2

        for _, bbox1 in outer_list:
            area1 = compute_area(bbox1)
            closest_chair = None
            min_area_diff = float('inf')

            for _, bbox2 in inner_list:
                area2 = compute_area(bbox2)
                area_diff = abs(area1 - area2)

                if area_diff < min_area_diff:
                    min_area_diff = area_diff
                    closest_chair = bbox2

            if closest_chair:
                matched_chairs.append((bbox1, closest_chair))
            

        #print(f"Pair of chair coordinates after matching: {matched_chairs}")
    
        return matched_chairs
    
    else:
        for _, box1 in chairs1:
            bbox1 = box1
        for _, box2 in chairs2:
            bbox2 = box2

        chair1_height = bbox1[3]
        chair2_height = bbox2[3]

        height_difference = abs(chair2_height - chair1_height)
        if height_difference < 8:
            matched_chairs.append((bbox1, bbox2))

        return matched_chairs

def compute_area(bbox):
    width = bbox[2]
    height = bbox[3]
    return width * height

def choose_largest_chair(matched_chairs):
    max_area = 0
    largest_pair = None

    for bbox1, bbox2 in matched_chairs:
        area1 = compute_area(bbox1)
        area2 = compute_area(bbox2)
        area = max(area1, area2)

        if area > max_area:
            max_area = area
            largest_pair = (bbox1, bbox2)
    print(f"largest pair is {largest_pair}")
    return largest_pair


def match_bins(bins1, bins2):
    min_distance = float('inf')
    matched_bins = None

    if len(bins1) > 1 or len(bins2) > 1:
        if len(bins1) < len(bins2):
            outer_list = bins1
            inner_list = bins2
        elif len(bins2) < len(bins1):
            outer_list = bins2
            inner_list = bins1
        else:
            outer_list = bins1
            inner_list = bins2

        for _, bbox1 in outer_list:
            center1 = get_center_coordinate(bbox1)
        
            for _, bbox2 in inner_list:
                center2 = get_center_coordinate(bbox2)
            
                # Calculate the Euclidean distance between the centers
                distance = np.linalg.norm(np.array(center1) - np.array(center2))
            
                if distance < min_distance:
                    min_distance = distance
                    matched_bins = (bbox1, bbox2)

        return matched_bins
    
    else:
        for _, bbox1 in bins1:
            height1 = bbox1[3]
        for _, bbox2 in bins2:
            height2 = bbox2[3]

        height_diff = abs(height2 - height1)

        if height_diff < 5:
            matched_bins = (bbox1, bbox2)

        return matched_bins
        
        

def get_center_coordinate(bbox):
    center_x = bbox[0]
    center_y = bbox[1]
    return (center_x, center_y)

def get_bounding_box_corners(bbox):
    x_center_pixel, y_center_pixel, width_pixel, height_pixel = bbox

    x_min = x_center_pixel - width_pixel // 2
    y_min = y_center_pixel - height_pixel // 2
    x_max = x_center_pixel + width_pixel // 2
    y_max = y_center_pixel + height_pixel // 2

    # Top-left corner
    top_left = (x_min, y_min)
    
    # Top-right corner
    top_right = (x_max, y_min)
    
    # Bottom-left corner
    bottom_left = (x_min, y_max)
    
    # Bottom-right corner
    bottom_right = (x_max, y_max)
    
    return top_left, top_right, bottom_right, bottom_left

def calculate_track_distance_and_centre(tracks):
    # Sort tracks by x-coordinate of their bounding box center
    tracks.sort(key=lambda x: get_center_coordinate(x[1])[0])

    # Extract bounding boxes for the left and right tracks from both images
    _, bbox_left = tracks[0]
    _, bbox_right = tracks[1]

    # Get the bottom left of the left track and bottom right of the right track for Image 1
    left_bbox_corners = get_bounding_box_corners(bbox_left)
    right_bbox_corners = get_bounding_box_corners(bbox_right)
    bottom_left_track1 = left_bbox_corners[3]  
    bottom_right_track2 = right_bbox_corners[2]  

    # Calculate the pixel wheelbase distance
    wheelbase_pixel_distance = abs(bottom_right_track2[0] - bottom_left_track1[0])
    y_distance = (bottom_right_track2[1] - bottom_left_track1[1])

    # Calculate the pixel wheelbase centre
    centre_x = int((wheelbase_pixel_distance/2) + bottom_left_track1[0])
    centre_y = int((y_distance/2) + bottom_left_track1[1])
    wheelbase_centre = (centre_x, centre_y)

    return wheelbase_pixel_distance, wheelbase_centre

def are_tracks_side_by_side(tracks):
    """
    Check if the tracks are side by side by comparing the x distance between their center pixels
    and the y alignment of the bottom left and bottom right corners.
    """
    if len(tracks) < 2:
        return False

    # Sort tracks by x-coordinate of their bounding box center
    tracks.sort(key=lambda x: get_center_coordinate(x[1])[0])

    # Get bounding boxes for the left and right tracks
    _, left_bbox = tracks[0]
    _, right_bbox = tracks[1]

    # Calculate the center coordinates
    left_center = get_center_coordinate(left_bbox)
    right_center = get_center_coordinate(right_bbox)

    # Calculate the x distance between the center pixels
    x_distance_centers = abs(right_center[0] - left_center[0])

    # Calculate the bottom corner coordinates
    left_bbox_corners = get_bounding_box_corners(left_bbox)
    right_bbox_corners = get_bounding_box_corners(right_bbox)
    bottom_left_left_track = left_bbox_corners[3]   
    bottom_right_right_track = right_bbox_corners[2]

    # Check if the y-coordinates are aligned
    y_aligned = abs(bottom_left_left_track[1] - bottom_right_right_track[1]) <= 10  # Adjust threshold as needed

    # Define threshold for x distance between centers
    x_distance_threshold = 340  # Threshold based on minimum observed x difference of 354.0

    is_track_seperate = x_distance_centers > x_distance_threshold

    # Check if the tracks are side by side
    return is_track_seperate
    
def read_relative_poses_from_csv(relative_pose_file):
    relative_poses = []
    timestamps = []
    with open(relative_pose_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            timestamps.append(row[0])
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            ox, oy, oz, ow = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            relative_poses.append((np.array([ox, oy, oz, ow]), np.array([tx, ty, tz])))
    return timestamps, relative_poses


def main():
    
    image_folder = "Data\RFC Pose Estimation Images"
    label_folder = "Data\Bounding_box_labels"
    index = 180
    is_scaled = 0
    
    # Data Storage for Pose data
    pose_data_file = os.path.join("Pose_Results", "ignore_RFC_relative_poses_approach5_5.csv")
    with open(pose_data_file, 'w') as f:
        f.write("pose_timestamp,tx,ty,tz,ox,oy,oz,ow,is_scaled,object_used\n")  # CSV header
    
    # Define Camera Instrinsics (Obtained from Zed Camera calibration file)
    K = np.array([[700.425, 0, 642.845],
                           [0, 700.425, 395.8775],
                           [0,0,1]])
    
    num_runs = 1 # Set number of runs (Total number needed for full run is 1639)

    for i in range(num_runs):
        # Load in the next pair of images and corresponding labels
        print(f"Index is for run {i} is {index}")
        im1, im2, label1, label2, index, ts1, ts2 = load_image_pair_and_labels(image_folder, label_folder, index) 
        #print (f"Image1 is {ts1} amd Image 2 is {ts2}")
        
        # Compute SIFT points and return matches using FLANN K-Nearest neighbour
        kp1, kp2, matches = detect_and_compute_points(im1, im2) 
        #print(f"Good matches for index {index-1}: {len(matches)}")

        # Compute the rotation matrix and translation vector from the essential matrix using RANSAC
        rotation_mat, translation_vec = compute_essential_matrix(kp1, kp2, matches, K)
   
        # Compute Scaled Translation vector
        vector_and_object_name = find_camera_translation_vector(im1, im2, label1, label2, K, rotation_mat)
             
        
        if vector_and_object_name:
            scaled_trans, object_used = vector_and_object_name

            # Find inverse of rotation matrix
            rot_mat_inv = rotation_mat.T

            # Compute inverse translation
            t_inverse = -rot_mat_inv @ scaled_trans

            tx = t_inverse[0] 
            ty = -t_inverse[1]
            tz = -t_inverse[2]

            t_scaled = np.asarray([(tx, ty, tz)])
            #t_scaled =np.asarray([(scaled_trans)])
            print(f"Translation of camera is {t_scaled})")
            #print(f"translation from Essential Matrix {translation_vec.T}")
    
            # Convert rotation matrix to quaternion
            rot_quat = convert_mat_to_quat(rot_mat_inv)
            
            # Flip the Y and Z component of the quaternion
            rot_quat[1] = -rot_quat[1]  # Flip y
            rot_quat[2] = -rot_quat[2] # Flip Z
                        
            # Reshape rotation and translation to perform concatenation
            rot_quat_reshaped = rot_quat.reshape(1, 4)
            print(f"Rotation Quarternion is: {rot_quat_reshaped}")
                        
            # Concatenate translation and quaternion into the final pose array
            final_pose = np.concatenate((t_scaled, rot_quat_reshaped), axis=1)

            # Set scaled translation flag to 1
            is_scaled = 1

            # Append the pose data to the CSV file
            with open(pose_data_file, "a") as f:
               f.write(f"{ts2}, {float(final_pose[0, 0]):.8f}, {float(final_pose[0, 1]):.8f}, {float(final_pose[0, 2]):.8f}, {float(final_pose[0, 3]):.8f}, {float(final_pose[0, 4]):.8f}, {float(final_pose[0, 5]):.8f}, {float(final_pose[0, 6]):.8f}, {is_scaled}, {object_used}\n")  

        else:
            print (f"No object found in run {i} for timestamp {ts2}, translation measurement set as zero")

            t_scaled = np.array([[0, 0, 0]])
            print(f"Translation of camera is {t_scaled})")

            # Find inverse of rotation matrix
            rot_mat_inv = rotation_mat.T

            # Convert rotation matrix to quaternion
            rot_quat = convert_mat_to_quat(rot_mat_inv)
            
            # Flip the Y and Z component of the quaternion
            rot_quat[1] = -rot_quat[1]  # Flip y
            rot_quat[2] = -rot_quat[2] # Flip Z
               
            # Reshape rotation and translation to perform concatenation
            rot_quat_reshaped = rot_quat.reshape(1, 4)
            print(f"Rotation Quarternion is: {rot_quat_reshaped}")
            
            # Concatenate translation and quaternion into the final pose array
            final_pose = np.concatenate((t_scaled, rot_quat_reshaped), axis=1)

            # Set scaled translation flag to 0 to notify particle filter to ignore this translation measurement and use commands only
            # Do not use translations with flag zero for calculating standard deviation with ground truth
            is_scaled = 0

            # Append the pose data to the CSV file
            with open(pose_data_file, "a") as f:
               f.write(f"{ts2}, {float(final_pose[0, 0]):.8f}, {float(final_pose[0, 1]):.8f}, {float(final_pose[0, 2]):.8f}, {float(final_pose[0, 3]):.8f}, {float(final_pose[0, 4]):.8f}, {float(final_pose[0, 5]):.8f}, {float(final_pose[0, 6]):.8f}, {is_scaled}, None\n")  
    
        
        
        
    print("Done computing Alternative raw poses")
       
    
if __name__ == '__main__':
    main()