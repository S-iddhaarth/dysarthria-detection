# Dysarthria Detection Project
## Introduction

Dysarthria is a motor speech disorder resulting from neurological injury that affects the muscles used in speech production, making it difficult for individuals to articulate words properly. Detecting dysarthria is crucial for early diagnosis and intervention.
Dataset

To work on this project, please download the Dysarthria dataset from Kaggle using the following link:

[Download Dataset](https://www.kaggle.com/datasets/pranaykoppula/torgo-audio)  
## Directory Structure

Once the dataset is downloaded, organize it in the following directory structure:  
```
Data/
└── dysarthria/
    ├── F_Con
    ├── F_Dys
    ├── M_Con
    ├── M_Dys
```
- ```F_Con```: Contains audio files of female control subjects.  
- ```F_Dys```: Contains audio files of female - dysarthric subjects.  
- ``` M_Con```: Contains audio files of male control subjects. 
- ```M_Dys```: Contains audio files of male dysarthric subjects.  

## Running the Script

After organizing the dataset, run the split.py script to preprocess the data.

bash
```
python split.py
```
This will prepare the data for further analysis and model training.

