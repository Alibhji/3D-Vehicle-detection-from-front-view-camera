# Deep Learning -Peking University/Baidu - Autonomous Driving (Kaggle Competition-3D object detection):
Competition link : [click](https://www.kaggle.com/c/pku-autonomous-driving)

There are two general ideas:
 - **Detect Center point of each object (x, y in image) then regress its properties (X, Y, Z, Roll, Yaw, Pitch in 3Dworld)**
 -- First we need genrate mask by using the center point of each object from annotation (Ground Truth) as you can see in following picture:
 -- In this method a heat map will be genrated for each object as you can see in the following image:
![enter image description here](./doc/images/center_of_objects_1.png)
