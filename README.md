# RefineSiamese
SiamFC is one the most popular tracking solutions in Short term tracking sub-track. Despite its popularity and promising results, it tends to fail during the scenarios of heavy clutter due its inability to properly understand the foreground and background. To this end, we have integrated the concepts of RefineNet which claims to help the network in better understanding the background and foreground. As a result, we achieved promising correlation output maps.


## Prerequisities
The solution was developed using MatConvNet and hence to develop/run you need to install this module.

## Dataset
To train the network, you need to download ILSVRC2015 VID dataset and update the corresponding paths in the code.

## Training and Tracking
Once the dataset is downloaded and all the variables are updated with the paths, you can call the train_network.m or tracker.m to train the network or track a target.

## Acknowledgement
This project is done as part of my summer internship work at Indian Institute of Space Science and Technology, Trivandrum.
Thanks to my mentor, Dr. Deepak Mishra, Associate Professor in Avionics department, who has guided me and motivated me enough to understand and implement this solution.
