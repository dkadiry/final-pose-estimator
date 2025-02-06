import os
import cv2
import numpy as np

def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO labels and convert them to pixel coordinates."""
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x_min = int(x_center - (width / 2))
            y_min = int(y_center - (height / 2))
            x_max = int(x_center + (width / 2))
            y_max = int(y_center + (height / 2))

            # Top-left corner
            top_left = (x_min, y_min)
    
            # Top-right corner
            top_right = (x_max, y_min)
    
            # Bottom-left corner
            bottom_left = (x_min, y_max)
    
            # Bottom-right corner
            bottom_right = (x_max, y_max)

            # Append bounding box as (class_id, (x_min, y_min, x_max, y_max))
            bboxes.append((class_id, (top_left, top_right, bottom_left, bottom_right)))
    return bboxes

def get_center_coordinate(bbox):
    top_left, _, bottom_right, _ = bbox
    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2
    return (center_x, center_y)

def calculate_differences(bboxes):
    """Calculate the x and y differences between two tracks."""
    if len(bboxes) < 2:
        return None, None

    centers = [get_center_coordinate(bbox[1]) for bbox in bboxes]
    
    # Assuming the two tracks are the first two bounding boxes (adjust as needed)
    x_diff = abs(centers[0][0] - centers[1][0])
    y_diff = abs(centers[0][1] - centers[1][1])

    # Sort tracks by x-coordinate of their bounding box center
    bboxes.sort(key=lambda x: get_center_coordinate(x[1])[0])

    # Get bounding boxes for the left and right tracks
    _, left_bbox = bboxes[0]
    _, right_bbox = bboxes[1]

    # Calculate the center coordinates
    left_center = get_center_coordinate(left_bbox)
    right_center = get_center_coordinate(right_bbox)

    # Calculate the x distance between the center pixels
    x_distance_centers = abs(right_center[0] - left_center[0])

    return x_distance_centers, y_diff

def main():
    # Directory paths
    img_dir = 'Data\data_for_tracks\images'
    label_dir = "Data\data_for_tracks\labels"
    x_diffs = []
    y_diffs = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            img_path = os.path.join(img_dir, label_file.replace('.txt', '.jpg'))

            # Load image to get its dimensions
            img = cv2.imread(img_path)
            img_height = img.shape[0]
            #print(img.shape)
            img_width = img.shape[1]

            # Load YOLO labels
            bboxes = load_yolo_labels(label_path, img_width, img_height)

            # Filter for track class id (adjust this to your specific class id for tracks)
            track_bboxes = [bbox for bbox in bboxes if bbox[0] == 9]

            # Calculate differences
            x_diff, y_diff = calculate_differences(track_bboxes)
            if x_diff is not None and y_diff is not None:
                x_diffs.append(x_diff)
                y_diffs.append(y_diff)
                print(f'X Difference: {x_diff}, Y Difference: {y_diff}')  # Print each x and y difference

    # Calculate averages and standard deviations
    if x_diffs and y_diffs:
        avg_x_diff = np.mean(x_diffs)
        std_x_diff = np.std(x_diffs)
        avg_y_diff = np.mean(y_diffs)
        std_y_diff = np.std(y_diffs)
        min_x_diff = np.min(x_diffs)
        max_x_diff = np.max(x_diffs)
        min_y_diff = np.min(y_diffs)
        max_y_diff = np.max(y_diffs)

        print(f'Average X Difference: {avg_x_diff}')
        print(f'Standard Deviation of X Difference: {std_x_diff}')
        print(f"Minimum X Difference is {min_x_diff}, Maximum X difference is {max_x_diff}")
        print(f'Average Y Difference: {avg_y_diff}')
        print(f'Standard Deviation of Y Difference: {std_y_diff}')
        print(f"Minimum Y Difference is {min_y_diff}, Maximum Y difference is {max_y_diff}")
    else:
        print('No valid track pairs found.')

if __name__ == '__main__':
    main()