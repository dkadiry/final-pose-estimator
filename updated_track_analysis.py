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

            # Append bounding box as (class_id, (x_min, y_min, x_max, y_max))
            bboxes.append((class_id, (x_min, y_min, x_max, y_max)))
    return bboxes

def calculate_wheelbase_distance(bboxes):
    """Calculate the wheelbase distance between two tracks."""
    if len(bboxes) < 2:
        return None

    # Sort bboxes by the x-coordinate of their center
    bboxes.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)

    # Get the bottom left of the left track and bottom right of the right track
    _, left_bbox = bboxes[0]
    _, right_bbox = bboxes[1]
    bottom_left_left_track = (left_bbox[0], left_bbox[3])   # (x_min, y_max)
    bottom_right_right_track = (right_bbox[2], right_bbox[3]) # (x_max, y_max)

    # Calculate the x distance (wheelbase)
    x_distance = bottom_right_right_track[0] - bottom_left_left_track[0]
    return x_distance

def main():
    # Directory paths
    img_dir = 'Data\data_for_tracks\images'  
    label_dir = 'Data\data_for_tracks\labels'  

    wheelbase_distances = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            img_path = os.path.join(img_dir, label_file.replace('.txt', '.jpg'))

            # Load image to get its dimensions
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Load YOLO labels
            bboxes = load_yolo_labels(label_path, img_width, img_height)

            # Filter for track class id (adjust this to your specific class id for tracks)
            track_bboxes = [bbox for bbox in bboxes if bbox[0] == 9]  # Adjust class_id if necessary

            # Calculate wheelbase distance
            x_distance = calculate_wheelbase_distance(track_bboxes)
            if x_distance is not None:
                wheelbase_distances.append(x_distance)
                print(f'Wheelbase Distance: {x_distance}')  # Print each wheelbase distance

    # Calculate averages and standard deviations
    if wheelbase_distances:
        avg_x_distance = np.mean(wheelbase_distances)
        std_x_distance = np.std(wheelbase_distances)
        min_x_distance = np.min(wheelbase_distances)
        max_x_distance = np.max(wheelbase_distances)

        print(f'\nAverage Wheelbase Distance: {avg_x_distance}')
        print(f'Standard Deviation of Wheelbase Distance: {std_x_distance}')
        print(f'Minimum Wheelbase Distance: {min_x_distance}')
        print(f'Maximum Wheelbase Distance: {max_x_distance}')
    else:
        print('No valid track pairs found.')

if __name__ == '__main__':
    main()
