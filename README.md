# DSS-Master-Thesis-Deepfake-detection
This repository contains the python scripts used for my Data Science &amp; Society Master thesis at Tilburg Univeristy

In this repository you will find three python scripts:

- frame_extraction.py
- facial_region_extraction.py
- Xception_model.ipynb

frame_extraction.py is the script used to extract frames from the datasets downloaded.
This file is a modified version of HaydenFaulkner's code. Original available at: 
https://gist.github.com/HaydenFaulkner/3aa69130017d6405a8c0580c63bee8e6

facial_region_extraction.py used the .csv files exported by the docker solution of OpenFace and extract different facial features.

Xception_model.py is the file used in Google Colab to train and evaluate the models proposed in the thesis.

