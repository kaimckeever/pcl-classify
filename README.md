# pcl-classify

## The repository

Full results and figures are available in their respective folders.

## Preparing to run the code

If you're a member of the CS 4650 teaching staff, the data for the PCL can be accessed from the Dropbox link submitted with the final report. Add the /data folder to the root of this repository.

If you're researcher from another institution, you can request access to the SemEval 2022 Task 4 dataset via a form provided by the organizers: https://forms.gle/VN8hwbdGYkf5KHiKA

## Running the code

A requirements.txt file has been provided. The jupyter notebook is also set up for pip installing packages not part of google colab, so any packages not already installed in the requirements.txt will be pip installed in the notebook.

The jupyter notebook file, CS_4650_Final_Project.ipynb, can be run in its entirety and will demonstrate SVM models as well as the train and validation loop for DistilBERT or BERT. 

CS_4650_Final_Project.ipynb can be uploaded to google colab for training on a GPU.

If uploaded to colab, the project will require preprocess.py for the SVMs.
