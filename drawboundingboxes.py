from PIL import Image, ImageDraw
import utils.tools as tools
from old_estimators import poseestimatorv4


index = 180
image_folder = "Data\RFC Pose Estimation Images"
label_folder = "Data\Bounding_box_labels"

#image_folder = "Data\\test\\images"
#label_folder = "Data\\test\\labels"

image1, image2, label1, label2, index, ts1, ts2, im1_path, im2_path = tools.load_image_pair_and_labels(image_folder, label_folder, index) 
bounding_boxes = []
                
for label in label1:
    class_id = int(label.split()[0])
    bbox = poseestimatorv4.get_bounding_box_pixels(label, 720, 1280)
    bounding_boxes.append((bbox))

#image_path = 'Data/RFC Pose Estimation Images/1717696051836.jpg'
image = Image.open(im1_path)
draw = ImageDraw.Draw(image)



    # Draw each bounding box
for box in bounding_boxes:
    x_center, y_center, box_width, box_height = box
    
    # Calculate the top-left and bottom-right coordinates
    top_left_x = x_center - box_width // 2
    top_left_y = y_center - box_height // 2
    bottom_right_x = x_center + box_width // 2
    bottom_right_y = y_center + box_height // 2
    
    # Draw the rectangle (bounding box) on the image
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="red", width=2)

    #text = f"(Top left),(Bottom Right) => ({top_left_x}, {top_left_y}), ({bottom_right_x}, {bottom_right_y})"
    text2 = f"(X_center, Y_center, width, height) => ({x_center}, {y_center}, {box_width}, {box_height})"

    draw.text((top_left_x, top_left_y - 10), text2, fill="yellow")  # You can also use `font=font` if a custom font is used
    #draw.text((bottom_right_x, bottom_right_y + 10), text2, fill="blue")

    # Display the image with bounding boxes
    #image.show()

    # Optionally, save the image with bounding boxes drawn
    output_path = f"Bounding_Box_Images/bb{index-1}_image{ts1}.jpg"
    image.save(output_path)