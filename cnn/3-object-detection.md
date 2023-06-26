# 3. Object Detection

## Problem


## Localization
How to localize an object in an image?
- bounding box (x, y, w, h), where (x, y) is the center of the box, w is the width and h is the height

target label design:
- $(p_c, x, y, w, h, c)$, where $p_c$ is the probability of an object, $c$ is the class of the object. $c$ could be a one-hot vector, whose length is the number of classes.


## Object Detection

there are many approaches to object detection, such as:
- sliding window
- region proposal

### Sliding Window
- slide a window across the image

Sliding window can be implemented by convolutional neural network (CNN), which is called **Convolutionalized**, to avoid redundant computation due to overlapping windows.

### Region Proposal
- generate a set of bounding boxes
- classify each bounding box

## YOLO

### Bounding Box Prediction

### Anchor Boxes 

### Intersection over Union (IoU)

### Non-max Suppression

### YOLO Algorithm

