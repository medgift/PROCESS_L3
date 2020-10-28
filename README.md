# Layer 3 of PROCESS Use Case 1
Layer III of UC1 PROCESS: Performance boosting and interpretability. 
(under development)

The use case tackles cancer detection and tissue classification on the latest challenges in cancer research using histopathology images from CAMELYON 16 and 17. 

# CAMNET: a three-layered software architecture
The software implemented by the use case consists of three layers. 
L 1. Data extraction and preprocessing: https://github.com/medgift/PROCESS_L1
L 2. Network training
L 3. Network interpretability 

# Dependencies
The code is written in Python 2.7 and requires Keras 2.1.5 with Tensorflow 1.4.0 as backend. Further dependencies are in doc/requirements.txt.

# Configuration
Configuration files are ini-based. A full template is in doc/config.cfg.

# Usage

The master script is a pipeline-based program that can be run by the command

python DHeatmaps.py 

This will distribute inference on the GPU by creating multiple instances of the deep learning model. 
The output of the script is saved in results/ and it is a heatmap of salient regions. 



