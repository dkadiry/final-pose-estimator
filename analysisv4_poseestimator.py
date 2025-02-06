import utils.tools as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import csv
import os
from statistics import mean
import math
from PIL import Image, ImageDraw
from sklearn.metrics import mean_absolute_error


def detect_and_compute_points(image1, image2):
    # Initialize sift detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    if descriptors_1 is not None and descriptors_2 is not None:
        # Create a FLANN based matcher and match descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Apply the ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return keypoints_1, keypoints_2, good_matches

def compute_essential_matrix(keypoints1, keypoints2, matches, cam_matrix):
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros_like(points1)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find Essential Matrix
    E, mask = cv2.findEssentialMat(points1, points2, cam_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover relative pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, cam_matrix)

    return R, t

def convert_mat_to_quat(rotation_mat):
    r = R.from_matrix(rotation_mat)
    rotation_quaternion = r.as_quat()
   
    return rotation_quaternion

def convert_quat_to_mat(rotation_quat):
    quart_obj = R.from_quat(rotation_quat)
    reconstructed_mat = quart_obj.as_matrix()

    return reconstructed_mat


def find_object_vector(image1, image2, labels1, labels2):
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
    k_original = 237.28
    horizontal_k_original = 323.46
    
    for label in labels1:
        class_id = int(label.split()[0])
        bbox = get_bounding_box_pixels(label, image1.shape[0], image1.shape[1])
        bboxes1.append((class_id, bbox))

    for label in labels2:
        class_id = int(label.split()[0])
        bbox = get_bounding_box_pixels(label, image2.shape[0], image2.shape[1])
        bboxes2.append((class_id, bbox))

    # Check for chairs first
    chairs1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == chair_class_id]
    chairs2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == chair_class_id]

    if chairs1 and chairs2:
        matched_chairs = match_chairs(chairs1, chairs2)
        if matched_chairs:
            chair_height_meters = 0.9
            k_chair = k_original * (chair_height_meters/0.31)
            #print(f"Chair k is {k_chair} ")
            chair1, chair2 = choose_largest_chair(matched_chairs)
            object_vector = compute_vector(k_chair, chair_height_meters, chair1, chair2)
            print(f"Object used is chair")
            return object_vector
        else:
            print("No matching chairs found between the two images.")

    # Check for other objects if no chairs found
    print("No chair found in one or both images, looking for alternative objects")
    filtered_bboxes1 = filter_classes(bboxes1)
    filtered_bboxes2 = filter_classes(bboxes2)

    if filtered_bboxes1 and filtered_bboxes2:
        common_classes = find_common_object(filtered_bboxes1, filtered_bboxes2)

        if common_classes:
            alt_object1, alt_object2 = find_alternative_object(filtered_bboxes1, filtered_bboxes2, common_classes)
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
                    k_object = k_original * (object_height_meters/0.31)
                    object_vector = compute_vector(k_object, object_height_meters, alt_bbox1, alt_bbox2)
                    print(f"Object is {object_name}")
                    return object_vector

    # Check for recycle bins if no other objects found
    print("No bags, monitor, box, or log found. Checking for recycle bins...")
    recycle_bins1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == recycle_box_id]
    recycle_bins2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == recycle_box_id]

    if recycle_bins1 and recycle_bins2:
        matched_bins = match_bins(recycle_bins1, recycle_bins2)
        if matched_bins:
            recycle_box1, recycle_box2 = matched_bins
            bin_height_meters = 0.33  # Real-world height of recycle bin in meters
            k_recycle = k_original * (bin_height_meters/0.31)
            object_vector = compute_vector(k_recycle, bin_height_meters, recycle_box1, recycle_box2)
            print(f"Object is recycle bin")
            return object_vector

    # Check for tracks if no recycle bins found
    print("No recycle bin found or bins do not match. Checking for tracks...")
    tracks1 = [(class_id, bbox) for class_id, bbox in bboxes1 if class_id == track_class_id]
    tracks2 = [(class_id, bbox) for class_id, bbox in bboxes2 if class_id == track_class_id]

    if len(tracks1) >= 2 and len(tracks2) >= 2:
        if are_tracks_side_by_side(tracks1) and are_tracks_side_by_side(tracks2):
            distance_im1, centre_im1 = calculate_track_distance_and_centre(tracks1)
            distance_im2, centre_im2 = calculate_track_distance_and_centre(tracks2)
            real_world_distance = 0.67  # Real-world distance between tracks in meters
            k_track = horizontal_k_original * (real_world_distance/0.454)

            # Find mtp ratio based on width
            mtp_ratio_x1 = real_world_distance / distance_im1
            mtp_ratio_x2 = real_world_distance / distance_im2

            # Calculate scaling factor for MTP_ratio based on camera FOV specs
            tan_fov_v_half = math.tan(math.radians(54.4 / 2))
            tan_fov_h_half = math.tan(math.radians(84.8 / 2))
            fov_ratio = tan_fov_h_half / tan_fov_v_half

            # Adjust the meters per pixel ratio for y-axis based on FOV
            mtp_ratio_y1 = mtp_ratio_x1 * fov_ratio
            mtp_ratio_y2 = mtp_ratio_x2 * fov_ratio
            
            # Find x and y displacement in real world based on mtp ratio from both images
            world_x_disp1 = (centre_im2[0] - centre_im1[0]) * mtp_ratio_x1
            world_x_disp2 = (centre_im2[0] - centre_im1[0]) * mtp_ratio_x1
            world_y_disp1 = (centre_im2[1] - centre_im1[1]) * mtp_ratio_y1
            world_y_disp2 = (centre_im2[1] - centre_im1[1]) * mtp_ratio_y2

            # Compute average displacements
            avg_x_disp = (world_x_disp1 +  world_x_disp2 ) / 2
            avg_y_disp = (world_y_disp1 +  world_y_disp2) / 2

            # Find distances to object in both images using slope k from experiment, where H = k / D => D = k / H
            z_track1 = k_track / (distance_im1 + 2.0111)
            z_track2 = k_track / (distance_im2 + 2.0111)
            ov_z = z_track2 - z_track1
            ov_x = avg_x_disp
            ov_y = avg_y_disp

            object_vector_norm = math.sqrt(ov_x**2 + ov_y**2 + ov_z**2)
            object_vector = ov_x, ov_y, ov_z
            print(f"Object used is Track")
            return object_vector
        print("Tracks found but they are not side by side")

    # Final else to catch all other cases
    print("No objects found for scale factor calculation")
    return None

def compute_vector(k, height_meters, bbox1, bbox2, fov_h = 84.8, fov_v = 54.4):
    # Find Object Height
    object_height_1 = bbox1[3]
    object_height_2 = bbox2[3]

    # Find object center
    object_center_1 = bbox1[0], bbox1[1]
    object_center_2 = bbox2[0], bbox2[1]

    # Find meters/pixel ratio at both depths based on height
    mtp_ratio_y1 = height_meters/object_height_1
    mtp_ratio_y2 = height_meters/object_height_2

    # Calculate scaling factor for MTP_ratio based on camera FOV specs
    tan_fov_v_half = math.tan(math.radians(fov_v / 2))
    tan_fov_h_half = math.tan(math.radians(fov_h / 2))
    fov_ratio = tan_fov_v_half / tan_fov_h_half
    print(f"FOV ratio is {fov_ratio}")

    # Find ratio from calibratio data
    ratio = 237.28/323.46
    #print(f"Ratio from Z calib data is {ratio}")

    # Adjust the meters per pixel ratio for x-axis based on FOV
    mtp_ratio_x1 = mtp_ratio_y1 * fov_ratio
    mtp_ratio_x2 = mtp_ratio_y2 * fov_ratio
   
    # Find distances to object in both images using slope k from experiment, where H = k / D => D = k / H
    z_object1 = k / (object_height_1 + 3.1709)
    z_object2 = k / (object_height_2 + 3.1709)
   
    # Compute x and y displacements in world units for each image
    x_disp_world1 = (object_center_2[0] - object_center_1[0]) * mtp_ratio_x1
    x_disp_world2 = (object_center_2[0] - object_center_1[0]) * mtp_ratio_x2
    y_disp_world1 = (object_center_2[1] - object_center_1[1]) * mtp_ratio_y1
    y_disp_world2 = (object_center_2[1] - object_center_1[1]) * mtp_ratio_y2

    # Compute average displacements
    avg_x_disp = (x_disp_world1 +  x_disp_world2) / 2
    avg_y_disp = (y_disp_world1 +  y_disp_world2) / 2

    # Find all three components of object vector for translation
    ov_z = z_object2 - z_object1
    ov_x = avg_x_disp
    ov_y = avg_y_disp

    #print(f"Object K is {k}")
    #print(f"Object Center 2 is {object_center_2} and Object center 1 is {object_center_1}")
    #print (f"Pixel x disp is {float(object_center_2[0] - object_center_1[0])}")
    #print (f"Object height in 1 is {object_height_1}")
    #print (f"Object height in 2 is {object_height_2}")
    #print (f"Bounding Box 1 in pixels is {bbox1}")
    #print (f"Bounding Box 2 in pixels is {bbox2}")
    #print(f"MTP Ratio_Y 1 is {mtp_ratio_y1}, MTP Ratio_Y 2 is {mtp_ratio_y2}")
    #print(f"X disp 1 is {x_disp_world1}, X disp 2 is {x_disp_world2}")
    #print(f"Distance to Object in Im1 is {z_object1}")
    #print(f"Distance to Object in Im 2 is {z_object2}")

    #scale_factor = math.sqrt(sf_x**2 + sf_y**2 + sf_z**2)
    object_vector = (ov_x, ov_y, ov_z)
    #print(f"Object Translation Vector is {object_vector}")

    return object_vector


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

    for _, bbox1 in chairs1:
        area1 = compute_area(bbox1)
        closest_chair = None
        min_area_diff = float('inf')

        for _, bbox2 in chairs2:
            area2 = compute_area(bbox2)
            area_diff = abs(area1 - area2)

            if area_diff < min_area_diff:
                min_area_diff = area_diff
                closest_chair = bbox2

        if closest_chair:
            matched_chairs.append((bbox1, closest_chair))


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

    return largest_pair


def match_bins(bins1, bins2):
    min_distance = float('inf')
    matched_bins = None

    for _, bbox1 in bins1:
        center1 = get_center_coordinate(bbox1)
        
        for _, bbox2 in bins2:
            center2 = get_center_coordinate(bbox2)
            
            # Calculate the Euclidean distance between the centers
            distance = np.linalg.norm(np.array(center1) - np.array(center2))
            
            if distance < min_distance:
                min_distance = distance
                matched_bins = (bbox1, bbox2)

    return matched_bins

def get_center_coordinate(bbox):
    center_x = bbox[0]
    center_y = bbox[1]
    return (center_x, center_y)

def calculate_track_distance_and_centre(tracks):
    # Sort tracks by x-coordinate of their bounding box center
    tracks.sort(key=lambda x: get_center_coordinate(x[1])[0])

    # Extract bounding boxes for the left and right tracks from both images
    _, bbox_left = tracks[0]
    _, bbox_right = tracks[1]

    # Get the bottom left of the left track and bottom right of the right track for Image 1
    bottom_left_track1 = bbox_left[3]  
    bottom_right_track2 = bbox_right[2]  

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

    # Calculate the bottom corner y-coordinates
    bottom_left_left_track = left_bbox[3]   
    bottom_right_right_track = right_bbox[2]

    # Check if the y-coordinates are aligned
    #y_aligned = abs(bottom_left_left_track[1] - bottom_right_right_track[1]) <= 10  # Adjust threshold as needed

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
    index = 0  
    scale_factors = []
    prev_scale_factor = 0
    is_scaled = 0
    directions = []
    scale_factors_x = []
    scale_factors_y = []
    scale_factors_z = []
    # Data Storage for Pose data
    pose_data_file = os.path.join("Pose_Results", "new_relative_poses.csv")
    with open(pose_data_file, 'w') as f:
        f.write("pose_timestamp,tx,ty,tz,ox,oy,oz,ow,is_scaled,scale_factor_used\n")  # CSV header
    
    # Define Camera Instrinsics (Obtained from Zed Camera calibration file)
    K = np.array([[700.425, 0, 642.845],
                           [0, 700.425, 395.8775],
                           [0,0,1]])
    
    num_runs = 1 # Set number of runs (Total number needed for full run is 1639)

    for i in range(num_runs):
        # Load in the next pair of images and corresponding labels
        image1, image2, label1, label2, index, _, ts2 = tools.load_image_pair_and_labels(image_folder, label_folder, index) 

        # Undistort images
        im1, im2 = tools.undistort_images(image1, image2)

        # Compute SIFT points and return matches using FLANN K-Nearest neighbour
        kp1, kp2, good_matches = detect_and_compute_points(im1, im2) 

        # Compute the rotation matrix and translation vector from the essential matrix using RANSAC
        rotation_mat, translation_vec = compute_essential_matrix(kp1, kp2, good_matches, K) 

        # Compute Object Vector and scale Translation vector
        object_vector = find_object_vector(im1, im2, label1, label2)
        #object_vector = np.asarray(object_vector)
        #object_vector_normal = (np.linalg.norm(object_vector))
        #object_vector_normalized = object_vector * (1/ object_vector_normal)
        
        #translation_vec = np.asarray(translation_vec).T
        # Transpose the Unit tarnslation vector from the Essential Matrix to (1, 3)
        t_transposed = translation_vec.T

        # Find the normal based on each of the components from the object vector
        sf_x = abs(object_vector[0]/t_transposed[0][0])
        sf_y = abs(object_vector[1]/t_transposed[0][1])
        sf_z = abs(object_vector[2]/t_transposed[0][2])

        # Find the normal based on the average from x and y
        #sf_xy = (sf_x + sf_y) / 2 

        print(f"Essential Matrix Unit Vector is {t_transposed}")
        #print(f"Normal from X {sf_x}, Y {sf_y}, Z {sf_z}, X and Y {sf_xy}")

        scale_factors_x.append(sf_x)
        scale_factors_y.append(sf_y)
        
        #print(F"Ty {t_transposed[0][1]}")
        #print(f"Ox is {object_vector[1]}")

        #t_scaled = object_vector_normal * translation_vec
        #print(f"Normal is {object_vector_normal}")
        print(f"Object Vector is {object_vector}")
        #print(f"Normalized Object translation Vector is {object_vector_normalized}")

        
        #print(f"Essential Matrix Scaled Vector is {t_scaled}")
        

        #test_direction = np.sum(object_vector_normalized * translation_vec)
        #directions.append(test_direction)
        #print(f" Direction is {test_direction}")

        
        
        """
        if scale_factor:
            scale_factors.append(scale_factor)
            prev_scale_factor = scale_factor
            print(f"Scale factor for run {i+1} and timestamp {ts2}: {scale_factor}")
            t_scaled = scale_factor * translation_vec

            
    
            # Convert rotation matrix to quaternion
            rot_quat = convert_mat_to_quat(rotation_mat)
    
            # Reshape rotation and translation to perform concatenation
            rot_quat_reshaped = rot_quat.reshape(1, 4)
            t_transposed = t_scaled.T

            # Concatenate translation and quaternion into the final pose array
            final_pose = np.concatenate((t_transposed, rot_quat_reshaped), axis=1)

            # Set scaled translation flag to 1
            is_scaled = 1

            # Append the pose data to the CSV file
            with open(pose_data_file, "a") as f:
               f.write(f"{ts2}, {float(final_pose[:, 0]):.8f}, {float(final_pose[:, 1]):.8f}, {float(final_pose[:, 2]):.8f}, {float(final_pose[:, 3]):.8f}, {float(final_pose[:, 4]):.8f}, {float(final_pose[:, 5]):.8f}, {float(final_pose[:, 6]):.8f}, {is_scaled}, {scale_factor}\n")  

        else:
            print (f"No scale factor computed in run {i+1} for timestamp {ts2}, Using previous Scale factor = {prev_scale_factor}")

            t_scaled = prev_scale_factor * translation_vec
    
            # Convert rotation matrix to quaternion
            rot_quat = convert_mat_to_quat(rotation_mat)
    
            # Reshape rotation and translation to perform concatenation
            rot_quat_reshaped = rot_quat.reshape(1, 4)
            t_transposed = t_scaled.T

            # Concatenate translation and quaternion into the final pose array
            final_pose = np.concatenate((t_transposed, rot_quat_reshaped), axis=1)

            # Set scaled translation flag to 1
            is_scaled = 0

            # Append the pose data to the CSV file
            with open(pose_data_file, "a") as f:
               f.write(f"{ts2}, {float(final_pose[:, 0]):.8f}, {float(final_pose[:, 1]):.8f}, {float(final_pose[:, 2]):.8f}, {float(final_pose[:, 3]):.8f}, {float(final_pose[:, 4]):.8f}, {float(final_pose[:, 5]):.8f}, {float(final_pose[:, 6]):.8f}, {is_scaled}, {prev_scale_factor}\n")  
    
    avg_asf = mean(scale_factors)
    std_asf = np.std(scale_factors)

    print(f"Average New Scale factor is {avg_asf}")
    print(f"Standard deviation of New scale factor is {std_asf}")

    print(f"Number of New scale factors is {len(scale_factors)}")
    print("Done computing Alternative raw poses")
        """

    #directions_arr = np.array(directions)

    #rmse = np.sqrt(np.mean((directions_arr - 1) ** 2))
    #print(f"RMSE is {rmse}")

    
    # Find MAE
    # Compute MAE directly using scikit-learn
    mae = mean_absolute_error(scale_factors_x, scale_factors_y)

    avg = (np.mean(scale_factors_x) + np.mean(scale_factors_y)) / 2

    
    #Calculate the relative differences, avoid division by zero
    rel_err = np.where(np.maximum(np.abs(scale_factors_x), np.abs(scale_factors_y)) == 0, 
                                0,  # Set relative difference to 0 when both values are 0
                                np.abs(np.array(scale_factors_x) - np.array(scale_factors_y)) / np.maximum(np.abs(scale_factors_x), np.abs(scale_factors_y)))


    # Calculate the mean relative difference
    mean_relative_difference = np.mean(rel_err)

    # Display
    print(f"MAE between normals from x and normals from y is {mae}")
    print(f"Average X factor is {np.mean(scale_factors_x)}")
    print(f"Average Y factors is {np.mean(scale_factors_y)}")
    print(f"Avg of both is {avg}")
    print(F"List of rel differences is {rel_err}")
    print(f"Average Relative Difference is {mean_relative_difference}")
        
    
if __name__ == '__main__':
    main()