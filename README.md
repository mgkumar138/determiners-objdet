## Repository to train models on Determiners dataset

### Installation
Install requirements using pip

```setup
pip install requirements.txt
```

### Get prediction 
To run the random bounding box selector model

```rand
python models/rand_bb_selector.py
```
To run the neural box selector model with none class

```
python models/ns_bb.py
```

To run the neural box selector model that predicts determiner class

```
python models/ns_bb_class.py
```

The scripts should store the ground truth and predictions in a separate folder as text files

### evaluate prediction

Change the path directory within the compute_map.py script to point to the folder containing ground truth and predictions
```
python models/compute_map.py
```

this will generate output folder within the initially created folder containing the mAP results.