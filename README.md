# Dysarthria Detection Project
## Introduction

Dysarthria is a motor speech disorder resulting from neurological injury that affects the muscles used in speech production, making it difficult for individuals to articulate words properly. Detecting dysarthria is crucial for early diagnosis and intervention.
Dataset

subhashni add the download process with the link to request the dataset

[Download Dataset](https://www.kaggle.com/datasets/pranaykoppula/torgo-audio)  
## Directory Structure

Once the dataset is downloaded and extracted the intial data directory will look like this:  
```
data/
└── dysarthria/
    ├── UASpeech_noisereduce_C
    ├── UASpeech_noisereduce_FM
        ├──audio
            ├──noisereduce
        ├──doc
        ├──mlf
    ├── UASpeech_normalized_C
    ├── UASpeech_normalized_FM-001
    ├── UASpeech_original_C
    ├── UASpeech_original_FM
```
In order to organize this run the following scripts

## Running the Script

first we need t organize the data directory


```bash
$ cd data_preparation
$ python organizeUA.py -d ..//data//dysarthria -o ..//data//UASPEECH
```
This will prepare the data for further processing.

we need to pair the dysarthric speech with controlled speach and create a csv file with annotation in order for easy data loading. This can be done by 

```bash
$ cd data_preparation
$ python .\\annotate.py -d ..\\data\\UASPEECH -t noisereduce -pm CM01 -pf CF02
```
you can substitute the -pm and -pf with the person id you want. To know more ways to se this module refer to the modules docstring
