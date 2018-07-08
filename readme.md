## Conceptual shifts model

This repository contains the code for identifying and selecting conceptual shifts among sketch categories. 


## Preparing the data

We use the binary format of the Quick Draw dataset available in: [Quick Draw!](https://github.com/googlecreativelab/quickdraw-dataset)

The original data has 5 objects: key-id, countrycode, recognized, timestamp and the image. For the purpose of classification, we use only the three elements including key-id which is the name of the category, recognized which indicates if the drawn sketch is recognized as belonging to its category or not and the image which is the coordinates of the drawing.

To generate the list of imags for a category, use:

    python read.py  --class-name  bridge   --dataset_path  bin-files/full-binary-bridge.bin

## Extracting features

We use a pre-trained CNN model, VGG-16, to extract features per category. The model is originally trained on ImageNet. We extract the features from the first fully connected layer, which corresponds to a 4096 feature-vector per sketch. 

To extract features for each category, use:

    Python features.py   --max_images   100000   --image_path  bridge

## Clustering

Cluster the sketches of each category into different subcategories using k-means algorithm. To determine the number of clusters, K, use the elbow method and plot the explained variance versus the number of clusters. It is observed that for most categories the appropriate value for K is between 7 to 12.   

To employ the elbow method and apply the k-means algorithm for each category, use:

'python elbow.py
python kmeans.py   --features_path   features/features_bridge.npy   --num_clusters   15'

We use LargeVis embedding (https://github.com/lferry007/LargeVis) to visualize the clusters in a 2D scatter plot. Output-nose.txt is the generated output for nose category using a random seed. In order to see sketches of each cluster, use:

'python visualize.embedding.py'

Note: The embedding showed here is different from the paper because we are using a different random seed. 

**Conceptual shifts results**

To see the potential conceptual shifts, we reported the results among 65 selected categories:

{'aircraft-carrier', 'airplane', 'alarm-clock', 'ambulance', 'angel', 'ant', 'apple', 'arm', 'axe', 'backpack', 'bananas', 'baseball', 'basketball', 'bathtub', 'beach', 'blackberry', 'brain', 'bridge' 'calculator', 'carrot', 'cat' ,'ceiling-fan', 'cell phone', 'church', 'computer', 'cookie', 'cooler', 'crocodile', 'crown', 'dolphin', 'donut', 'dumbbell', 'eye', , 'eraser', 'finger', 'fish', 'flower', 'grapes', 'hamburger', 'hand', 'hotdog', 'ice cream', 'knee’, 'leg', 'mermaid', 'moustache', 'mouth', 'mushroom', 'nose', 'ocean', 'penguin', 'pineapple', 'potato', 'rainbow', 'roller-coaster', 'shark', 'sheep', 'smiley face', 'snail', 'strawberry', 'tooth', 'toothbrush', 'tree', 'turtle', 'zebra'} 

To see the potential shifted categories with the associated Euclidean distances for each, use:

'Python distance.py   --max_num_cs   1'

For example, the output for single input and single output (max_num_cs =1) will be:

[('19.412', ((5, 3), 'aircraft-carrier', 'crocodile'))]
[('21.142', ((2, 19), 'airplane', 'aircraft-carrier'))]
[('28.673', ((2, 6), 'alarm-clock', 'beach'))]
…
…
[('32.633', ((7, 13), 'dumbbell', 'baseball'))]

The first value is the Euclidean distance between the two clusters. For instance, the cluster 5 of aircraft-carrier is most similar to cluster 3 of crocodile with the distance of 19.412. 

For max_num_cs =5, the results for bridge will be:

[('27.832', ((1, 8), 'bridge', 'rainbow')), ('30.649', ((3, 1), 'bridge', 'beach')), ('33.531', ((3, 10), 'bridge', 'shark')), ('34.322', ((0, 3), 'bridge', 'church')), ('34.428', ((2, 16), 'bridge', 'roller-coaster'))]


