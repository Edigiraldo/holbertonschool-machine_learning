0x0A. Object Detection
======================

Learning Objectives
-------------------

### General

-   What is OpenCV and how do you use it?
-   What is object detection?
-   What is the Sliding Windows algorithm?
-   What is a single-shot detector?
-   What is the YOLO algorithm?
-   What is IOU and how do you calculate it?
-   What is non-max suppression?
-   What are anchor boxes?
-   What is mAP and how do you calculate it?


Tasks
-----

#### 0\. Initialize Yolo

Write a class `Yolo` that uses the Yolo v3 algorithm to perform object detection:

-   class constructor: `def __init__(self, model_path, classes_path, class_t, nms_t, anchors):`
    -   `model_path` is the path to where a Darknet Keras model is stored
    -   `classes_path` is the path to where the list of class names used for the Darknet model, listed in order of index, can be found
    -   `class_t` is a float representing the box score threshold for the initial filtering step
    -   `nms_t` is a float representing the IOU threshold for non-max suppression
    -   `anchors` is a `numpy.ndarray` of shape `(outputs, anchor_boxes, 2)` containing all of the anchor boxes:
        -   `outputs` is the number of outputs (predictions) made by the Darknet model
        -   `anchor_boxes` is the number of anchor boxes used for each prediction
        -   `2` => `[anchor_box_width, anchor_box_height]`
-   Public instance attributes:
    -   `model`: the Darknet Keras model
    -   `class_names`: a list of the class names for the model
    -   `class_t`: the box score threshold for the initial filtering step
    -   `nms_t`: the IOU threshold for non-max suppression
    -   `anchors`: the anchor boxes

#### 1\. Process Outputs

Write a class `Yolo` (Based on `0-yolo.py`):

-   Add the public method `def process_outputs(self, outputs, image_size):`
    -   `outputs` is a list of `numpy.ndarray`s containing the predictions from the Darknet model for a single image:
        -   Each output will have the shape `(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)`
            -   `grid_height` & `grid_width` => the height and width of the grid used for the output
            -   `anchor_boxes` => the number of anchor boxes used
            -   `4` => `(t_x, t_y, t_w, t_h)`
            -   `1` => `box_confidence`
            -   `classes` => class probabilities for all classes
    -   `image_size` is a `numpy.ndarray` containing the image's original size `[image_height, image_width]`
    -   Returns a tuple of `(boxes, box_confidences, box_class_probs)`:
        -   `boxes`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 4)` containing the processed boundary boxes for each output, respectively:
            -   `4` => `(x1, y1, x2, y2)`
            -   `(x1, y1, x2, y2)` should represent the boundary box relative to original image
        -   `box_confidences`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 1)` containing the box confidences for each output, respectively
        -   `box_class_probs`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, classes)` containing the box's class probabilities for each output, respectively

#### 2\. Filter Boxes

Write a class `Yolo` (Based on `1-yolo.py`):

-   Add the public method `def filter_boxes(self, boxes, box_confidences, box_class_probs):`
    -   `boxes`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 4)` containing the processed boundary boxes for each output, respectively
    -   `box_confidences`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, 1)` containing the processed box confidences for each output, respectively
    -   `box_class_probs`: a list of `numpy.ndarray`s of shape `(grid_height, grid_width, anchor_boxes, classes)` containing the processed box class probabilities for each output, respectively
    -   Returns a tuple of `(filtered_boxes, box_classes, box_scores)`:
        -   `filtered_boxes`: a `numpy.ndarray` of shape `(?, 4)` containing all of the filtered bounding boxes:
        -   `box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number that each box in `filtered_boxes` predicts, respectively
        -   `box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for each box in `filtered_boxes`, respectively

#### 3\. Non-max Suppression

Write a class `Yolo` (Based on `2-yolo.py`):

-   Add the public method `def non_max_suppression(self, filtered_boxes, box_classes, box_scores):`
    -   `filtered_boxes`: a `numpy.ndarray` of shape `(?, 4)` containing all of the filtered bounding boxes:
    -   `box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number for the class that `filtered_boxes` predicts, respectively
    -   `box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for each box in `filtered_boxes`, respectively
    -   Returns a tuple of `(box_predictions, predicted_box_classes, predicted_box_scores)`:
        -   `box_predictions`: a `numpy.ndarray` of shape `(?, 4)` containing all of the predicted bounding boxes ordered by class and box score
        -   `predicted_box_classes`: a `numpy.ndarray` of shape `(?,)` containing the class number for `box_predictions` ordered by class and box score, respectively
        -   `predicted_box_scores`: a `numpy.ndarray` of shape `(?)` containing the box scores for `box_predictions` ordered by class and box score, respectively

#### 4\. Load images

Write a class `Yolo` (Based on `3-yolo.py`):

-   Add the static method `def load_images(folder_path):`
    -   `folder_path`: a string representing the path to the folder holding all the images to load
    -   Returns a tuple of `(images, image_paths)`:
        -   `images`: a list of images as `numpy.ndarray`s
        -   `image_paths`: a list of paths to the individual images in `images`

#### 5\. Preprocess images

Write a class `Yolo` (Based on `4-yolo.py`):

-   Add the public method `def preprocess_images(self, images):`
    -   `images`: a list of images as `numpy.ndarray`s
    -   Resize the images with inter-cubic interpolation
    -   Rescale all images to have pixel values in the range `[0, 1]`
    -   Returns a tuple of `(pimages, image_shapes)`:
        -   `pimages`: a `numpy.ndarray` of shape `(ni, input_h, input_w, 3)` containing all of the preprocessed images
            -   `ni`: the number of images that were preprocessed
            -   `input_h`: the input height for the Darknet model *Note: this can vary by model*
            -   `input_w`: the input width for the Darknet model *Note: this can vary by model*
            -   `3`: number of color channels
        -   `image_shapes`: a `numpy.ndarray` of shape `(ni, 2)` containing the original height and width of the images
            -   `2` => `(image_height, image_width)`

#### 6\. Show boxes

Write a class `Yolo` (Based on `5-yolo.py`):

-   Add the public method `def show_boxes(self, image, boxes, box_classes, box_scores, file_name):`
    -   `image`: a `numpy.ndarray` containing an unprocessed image
    -   `boxes`: a `numpy.ndarray` containing the boundary boxes for the image
    -   `box_classes`: a `numpy.ndarray` containing the class indices for each box
    -   `box_scores`: a `numpy.ndarray` containing the box scores for each box
    -   `file_name`: the file path where the original image is stored
    -   Displays the image with all boundary boxes, class names, and box scores *(see example below)*
        -   Boxes should be drawn as with a blue line of thickness 2
        -   Class names and box scores should be drawn above each box in red
            -   Box scores should be rounded to 2 decimal places
            -   Text should be written 5 pixels above the top left corner of the box
            -   Text should be written in `FONT_HERSHEY_SIMPLEX`
            -   Font scale should be 0.5
            -   Line thickness should be 1
            -   You should use `LINE_AA` as the line type
        -   The window name should be the same as `file_name`
        -   If the `s` key is pressed:
            -   The image should be saved in the directory `detections`, located in the current directory
            -   If `detections` does not exist, create it
            -   The saved image should have the file name `file_name`
            -   The image window should be closed
        -   If any key besides `s` is pressed, the image window should be closed without saving

#### 7\. Predict

Write a class `Yolo` (Based on `6-yolo.py`):

-   Add the public method `def predict(self, folder_path):`
    -   `folder_path`: a string representing the path to the folder holding all the images to predict
    -   All image windows should be named after the corresponding image filename without its full path*(see examples below)*
    -   Displays all images using the `show_boxes` method
    -   Returns: a tuple of `(predictions, image_paths)`:
        -   `predictions`: a list of tuples for each image of `(boxes, box_classes, box_scores)`
        -   `image_paths`: a list of image paths corresponding to each prediction in `predictions`
