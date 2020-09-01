# Deep Learning -Peking University/Baidu - Autonomous Driving (Kaggle Competition-3D object detection):
Competition link : [click](https://www.kaggle.com/c/pku-autonomous-driving)

There are two general ideas: \
[**First is Center-point detections**]
 - **Detect Center point of each object (x, y in image) then regress its properties (X, Y, Z, Roll, Yaw, Pitch in 3Dworld)**\
 - First the masks will be generated by using the center point of each object from annotation (Ground Truth) as you can see in following pictures:\
 - center poitns:\
![enter image description here](./doc/images/center_of_objects_1.png)
- imags and coresponding generated masks: [**Orginal Image , 1x Center Point Mask, 6x Regresion Masks**]\
![enter image description here](./doc/images/center_of_objects_2.png)
![enter image description here](./doc/images/center_of_objects_3.png)
![enter image description here](./doc/images/center_of_objects_4.png)
- The following Struture is used to train the network:\
![enter image description here](./doc/images/model_center_2.png)

[**Second Key-point detection [corners of the object 3d cube]**]
- First the masks will be generated by using the 3D corners of each object that can be calculated from annotation (Ground Truth) as you can see in following pictures:\
![enter image description here](./doc/images/key_point1..png)
