# Object Recognition-Based Autonomous Driving in GTA 5

This project aims to perform autonomous driving tests using object recognition in the GTA 5 game. Below, you will find the explanation of the code, the libraries used, and how to download the necessary files.

## Features

- Real-time object recognition (humans, vehicles, traffic signs, etc.)
- Highlighting specific objects with different colors
- Calculating and displaying the distance of the nearest vehicle
- Screen capturing and processing

## Purpose

This project aims to demonstrate the applicability of object recognition algorithms in the GTA 5 game and how they can be used in autonomous driving tests. YOLOv3 is used as the object recognition model.

## Libraries and Installation Commands

The following libraries need to be installed for the project to work:

- OpenCV
- NumPy
- mss

Installation commands:

```bash
pip install opencv-python-headless numpy mss
```


## YOLOv3 Model Files
You need to download the YOLOv3 model files. Follow the steps below to download the files:

Download the yolov3.weights file from this link and place it in the project directory.
Download the yolov3.cfg and coco.names files from here and place them in the project directory as well.

## Code Explanation
Loading the YOLO Model

```
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = model.getLayerNames()
output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
```
This code loads the YOLO model files and determines the output layers.

## Labels and Colors
```
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = {
    "person": (0, 255, 255),  # Yellow
    "bicycle": (42, 42, 165),  # Brown
    "car": (0, 255, 0),  # Green
    ...
}
```
Labels are read from the coco.names file, and a color is assigned to each class.

## Screen Capture and Object Recognition
```
with mss.mss() as screen:
    monitor = screen.monitors[1]

    while True:
        screen_shot = screen.grab(monitor)
        image = np.array(screen_shot)
        ...
```
The screen is captured, and object recognition processes are performed.

## Distance Calculation and Visualization
```
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

if nearest_vehicle_coordinates is not None:
    vehicle_distance_text = f"Distance: {nearest_vehicle_distance / 10:.1f} m"
    cv2.line(image, (width // 2, height), nearest_vehicle_coordinates, (0, 255, 255), 2)
    cv2.putText(image, vehicle_distance_text, (nearest_vehicle_coordinates[0], nearest_vehicle_coordinates[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
```
The distances of objects are calculated, and the distance of the nearest vehicle is displayed.

## Running the Code
```
python gta5_autonomous_driving.py
```
While the code is running, the labels and colored boxes of objects will appear in the top right corner of the screen. The distance of the nearest vehicle will be displayed at the bottom of the screen.

## Acknowledgements
- YOLO - For the object detection algorithm.
- OpenCV - For the computer vision library.
- mss - For the screenshot library.


