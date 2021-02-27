#!/usr/bin/env python3
"""Class Yolo."""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses the Yolo v3 algorithm
    to perform object detection."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor.

           - model_path is the path to where a Darknet Keras
             model is stored.
           - classes_path is the path to where the list of class
             names used for the Darknet model, listed in order of
             index, can be found.
           - class_t is a float representing the box score threshold
             for the initial filtering step.
           - nms_t is a float representing the IOU threshold for
             non-max suppression.
           - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2).
             containing all of the anchor boxes:
               - outputs is the number of outputs (predictions)
                 made by the Darknet model.
               - anchor_boxes is the number of anchor boxes used
                 for each prediction.
               - 2 => [anchor_box_width, anchor_box_height].
        """
        custom_objects = {'GlorotUniform': K.initializers.glorot_uniform()}
        self.model = K.models.load_model(model_path, custom_objects)
        self.class_names = open(classes_path, 'r').read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Public method.

            - outputs is a list of numpy.ndarrays containing the
              predictions from the Darknet model for a single image:
                - Each output will have the shape (grid_height, grid_width,
                  anchor_boxes, 4 + 1 + classes).
                    - grid_height & grid_width => the height and width
                      of the grid used for the output.
                    - anchor_boxes => the number of anchor boxes used.
                    - 4 => (t_x, t_y, t_w, t_h).
                    - 1 => box_confidence.
                    - classes => class probabilities for all classes.
                    - image_size is a numpy.ndarray containing the image’s
                      original size [image_height, image_width].
        """
        processed = ([], [], [])
        all_anchor_sizes = self.anchors
        anchor = 0
        img_h = image_size[0]
        img_w = image_size[1]
        for output in outputs:
            anchor_sizes = all_anchor_sizes[anchor]
            anchor += 1

            boxes = np.zeros(output[:, :, :, 0:4].shape)
            boxes[:, :, :, :] = output[:, :, :, 0:4]
            box_confidences = sigmoid(output[:, :, :, np.newaxis, 4])
            box_class_probs = sigmoid(output[:, :, :, 5:])

            gh = output.shape[0]
            gw = output.shape[1]
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    cy = i
                    cx = j
                    boxes[i, j, :, 0] = (sigmoid(output[i, j, :, 0]) + cx) / gw
                    boxes[i, j, :, 1] = (sigmoid(output[i, j, :, 1]) + cy) / gh

            inp_h = self.model.input.shape[2].value
            inp_w = self.model.input.shape[1].value
            pw = anchor_sizes[:, 0]
            ph = anchor_sizes[:, 1]
            boxes[:, :, :, 2] = pw * np.exp(output[:, :, :, 2]) / inp_w
            boxes[:, :, :, 3] = ph * np.exp(output[:, :, :, 3]) / inp_h

            coordinates = np.zeros(boxes.shape)
            coordinates[:, :, :, :] = boxes[:, :, :, :]

            bx = boxes[:, :, :, 0]
            by = boxes[:, :, :, 1]
            bw = boxes[:, :, :, 2]
            bh = boxes[:, :, :, 3]

            coordinates[:, :, :, 0] = (bx - bw / 2) * img_w
            coordinates[:, :, :, 1] = (by - bh / 2) * img_h
            coordinates[:, :, :, 2] = (bx + bw / 2) * img_w
            coordinates[:, :, :, 3] = (by + bh / 2) * img_h

            processed[0].append(coordinates)
            processed[1].append(box_confidences)
            processed[2].append(box_class_probs)

        return processed

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Public method.

           - boxes: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 4) containing the processed
             boundary boxes for each output, respectively.
           - box_confidences: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 1) containing
             the processed box confidences for each output, respectively.
           - box_class_probs: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, classes) containing
             the processed box class probabilities for each output,
             respectively.

           - Returns a tuple of (filtered_boxes, box_classes, box_scores):
               - filtered_boxes: a numpy.ndarray of shape (?, 4)
                 containing all of the filtered bounding boxes.
               - box_classes: a numpy.ndarray of shape (?,) containing the
                 class number that each box in filtered_boxes predicts,
                 respectively.
               - box_scores: a numpy.ndarray of shape (?) containing the
                 box scores for each box in filtered_boxes, respectively.
        """
        all_filtered_boxes = []
        all_box_classes = []
        all_scores = []

        N_outs = len(boxes)
        for out_idx in range(N_outs):
            box = boxes[out_idx]
            box_confidence = box_confidences[out_idx]
            box_class_prob = box_class_probs[out_idx]

            obj_class_prob = box_class_prob * box_confidence
            box_classes = np.argmax(obj_class_prob, axis=-1)
            box_scores = np.max(obj_class_prob, axis=-1, keepdims=False)

            mask = box_scores >= self.class_t

            filtered_boxes = box[mask]
            box_classes = box_classes[mask]
            box_scores = box_scores[mask]

            all_filtered_boxes.extend(filtered_boxes.tolist())
            all_box_classes.extend(box_classes.tolist())
            all_scores.extend(box_scores.tolist())

        all_filtered_boxes = np.array(all_filtered_boxes)
        all_box_classes = np.array(all_box_classes)
        all_scores = np.array(all_scores)

        return all_filtered_boxes, all_box_classes, all_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Public method.

         - filtered_boxes: a numpy.ndarray of shape (?, 4) containing
           all of the filtered bounding boxes.
         - box_classes: a numpy.ndarray of shape (?,) containing the
           class number for the class that filtered_boxes predicts,
           respectively.
         - box_scores: a numpy.ndarray of shape (?) containing the box
           scores for each box in filtered_boxes, respectively.

         - Returns a tuple of (box_predictions, predicted_box_classes,
           predicted_box_scores):
             - box_predictions: a numpy.ndarray of shape (?, 4) containing
               all of the predicted bounding boxes ordered by class and
               box score.
             - predicted_box_classes: a numpy.ndarray of shape (?,)
               containing the class number for box_predictions ordered
               by class and box score, respectively.
             - predicted_box_scores: a numpy.ndarray of shape (?) containing
               the box scores for box_predictions ordered by class and box
               score, respectively.
        """
        sort_indexes = np.lexsort((-box_scores, box_classes))
        box_predictions = filtered_boxes[sort_indexes]
        p_box_classes = box_classes[sort_indexes]
        predicted_box_scores = box_scores[sort_indexes]
        _, count_class = np.unique(p_box_classes, return_counts=True)

        index_collect = 0
        i = 0
        for n in count_class:
            while i < index_collect + n:
                j = i + 1
                while j < index_collect + n:
                    iou = self.iou(box_predictions[i], box_predictions[j])
                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions, j, axis=0)
                        p_box_classes = np.delete(p_box_classes, j, axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        n = n - 1
                    else:
                        j = j + 1
                i = i + 1
            index_collect = index_collect + n

        return box_predictions, p_box_classes, predicted_box_scores

    @staticmethod
    def iou(boxA, boxB):
        """this is the iou of the box against all other boxes"""
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))
