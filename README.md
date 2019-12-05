# Python Traffic Counter

The purpose of this project is to detect and track vehicles on a video stream and count those going through a defined line. 

It uses:

* [YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

Once the objects are detected and tracked over different frames a simple mathematical calculation is applied to count the intersections between the vehicles previous and current frame positions with a defined line.

The code on this prototype uses the code structure developed by Adrian Rosebrock for his article [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv).

## Quick Start
Make sure that darkflow is installed on your computer as this project depends on that for object detection.
The YOLO weights are present in 'build_graph' and the TFmodel is present in 'cfg' folder that is used in this project to make the predictions

A Flask API was made so that the user just have to provide the path of the video and the coordinates of the region where the counting algorithm should work.
Initially, the 'flaskk.py' is called and then this file sequentially calls all the other methods.

## Citation

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }