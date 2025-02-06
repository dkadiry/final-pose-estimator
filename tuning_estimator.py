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
import time

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


def detect_and_compute_points(image1, image2, trees=5, checks=50, thresh=0.75):
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
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
        search_params = dict(checks=checks)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Apply the ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < thresh * n.distance:
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

    return R, t, mask

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
    #print(f"largest pair is {largest_pair}")
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


def evaluate_flann_params(image_folder, label_folder, param_combination, num_runs=100):
    index = 0
    K = np.array([[700.425, 0, 642.845], [0, 700.425, 395.8775], [0, 0, 1]])
    
    # Initialize metrics
    total_good_matches = 0
    total_inlier_ratio = 0
    total_time = 0
    
    for i in range(num_runs):
        # Load image pairs and labels
        im1, im2, label1, label2, index, ts1, ts2 = load_image_pair_and_labels(image_folder, label_folder, index)
        
        # Skip if there are no more image pairs
        if im1 is None or im2 is None:
            break
               
        # Start timing
        start_time = time.time()
        
        # Detect and compute keypoints
        kp1, kp2, good_matches = detect_and_compute_points(im1, im2, trees=param_combination['trees'], checks=param_combination['checks'], thresh=param_combination['thresh'])
        
        # Compute essential matrix and inliers
        if len(good_matches) > 0:
            R, t, mask = compute_essential_matrix(kp1, kp2, good_matches, K)

            # Calculate the inlier ratio based on the mask
            inliers = mask.ravel().sum()  # Count the inliers from the mask
            inlier_ratio = inliers / len(good_matches)  # Ratio of inliers to good matches
        else:
            inlier_ratio = 0

        # End timing
        match_time = time.time() - start_time

        # Aggregate metrics
        total_good_matches += len(good_matches)
        total_inlier_ratio += inlier_ratio
        total_time += match_time

    # Calculate average metrics
    avg_good_matches = total_good_matches / num_runs
    avg_inlier_ratio = total_inlier_ratio / num_runs
    avg_time = total_time / num_runs
    
    return avg_good_matches, avg_inlier_ratio, avg_time

def tuning_main():
    image_folder = "Data/RFC Pose Estimation Images"
    label_folder = "Data/Bounding_box_labels"
    
    # Define parameter grid
    param_grid = {
        'trees': [6, 8],
        'checks': [200, 400],
        'lowe_thresh': [0.6, 0.7, 0.75]
    }
    
    best_params = None
    best_score = float('-inf')
    max_good_matches = 0
    results = []

    print("Running Cross-Validation...")

    # Run cross-validation for each parameter combination
    for trees in param_grid['trees']:
        for thresh in param_grid['lowe_thresh']:
            for checks in param_grid['checks']:
                params = {'trees': trees, 'thresh': thresh, 'checks': checks}
                avg_good_matches, avg_inlier_ratio, avg_time = evaluate_flann_params(image_folder, label_folder, params, num_runs=100)

                if max_good_matches < avg_good_matches:
                    max_good_matches = avg_good_matches
                
                # Define a scoring metric (e.g., inlier ratio - time penalty)
                #score = avg_inlier_ratio  # Adjust the weight of time penalty as needed  {- 0.01 * avg_time} 

                results.append({
                    'params': params,
                    'avg_good_matches': avg_good_matches,
                    'avg_inlier_ratio': avg_inlier_ratio,
                    'avg_time': avg_time
                    
                })

                for result in results:
                    # Extract values
                    params = result['params']
                    avg_good_matches = result['avg_good_matches']
                    avg_inlier_ratio = result['avg_inlier_ratio']
                    avg_time = result['avg_time']
                    
                    # Normalize good matches and calculate score
                    normalized_good_matches = avg_good_matches / max_good_matches if max_good_matches > 0 else 0
                    score = avg_inlier_ratio + 0.5 * normalized_good_matches  # Adjust weights as needed

                    # Update best score and parameters if this score is higher
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    # Append final scoring results
                    result['score'] = score
    
    # Output the best parameters and their performance
    print("Best FLANN parameters:", best_params)
    print("Results from cross-validation:")
    for result in results:
        print(result)
    
       
    
if __name__ == '__main__':
    tuning_main()