'''
This module is used to parse through the dataset directory and pair the dysarthric and controlled speech
with required annotation and generate a csv file which can be used to load the data. example to run this file

>>> python .\\annotate.py -d ..\\data\\UASPEECH -t noisereduce -pm CM01 -pf CF02
    
    -d - dataset directory
    -t - type of data being used (ordinory,noisereduce,normalized)
    -pm - controlled male file that needs to be used for pairing
    -pf - controlled female file that needs to be used for pairing
    
    if you have already run this code once and generated a key_pair file then use this command
    to avoid redaundant calculation of key pair
>>> python .\\annotate.py -d ..\\data\\UASPEECH -t noisereduce -pm CM01 -pf CF02 -pk .\\pair_key.json
'''

import os
import pandas as pd
from torch.utils.data import Dataset
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import csv

def get_metadata(root: str, data_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """this functon returns the dataframe containing intelligibility information
    and word id - word pair as a tuple

    Args:
        root (str): dataset directory
        data_type (str): type of data (ordinory,noisereduce,normalized)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: _description_
    """    
    word_path = os.path.join(root, "metadata", data_type, "doc", "speaker_wordlist.xls")
    annotations = pd.read_excel(word_path, sheet_name=None)
    word_list = annotations['Word_filename']
    new = annotations['Speaker'].dropna().reset_index()
    new.columns = new.iloc[0]
    intelligibility = new.drop(columns=2, index=0)

    return word_list, intelligibility

def get_sample_path(dataset:str, pair_male_id:str, wordID:str, microphone:str)->tuple:
    
    req = os.path.join(dataset, "controlled", pair_male_id, f'{pair_male_id}*{microphone}*')
    return wordID, microphone, glob.glob(req)[0]

def generate_pair_key(word_list, dataset, pair_male_id, pair_female_id):
    microphones = ["M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    
    pair_key = {
        'male': {},
        'female': {}
    }
    
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for word, wordID in tqdm(word_list.itertuples(index=False), total=len(microphones) * len(word_list), desc="Loading controlled (Male)"):
            for microphone in microphones:
                futures.append(executor.submit(get_sample_path, dataset, pair_male_id, wordID, microphone))
        
        for future in tqdm(futures, desc="Processing male results"):
            wordID, microphone, path = future.result()
            if wordID not in pair_key['male']:
                pair_key['male'][wordID] = {}
            pair_key['male'][wordID][microphone] = path

        futures = []
        
        for word, wordID in tqdm(word_list.itertuples(index=False), total=len(microphones) * len(word_list), desc="Loading controlled (Female)"):
            for microphone in microphones:
                futures.append(executor.submit(get_sample_path, dataset, pair_female_id, wordID, microphone))
        
        for future in tqdm(futures, desc="Processing female results"):
            wordID, microphone, path = future.result()
            if wordID not in pair_key['female']:
                pair_key['female'][wordID] = {}
            pair_key['female'][wordID][microphone] = path

    return pair_key

def process_row(personID, word, wordID, dataset, pair_key_value):
    rows = []
    required = os.path.join(dataset, "dysarthria", personID[0], f'{personID[0]}*{wordID}*')
    required_files = glob.glob(required)
    
    already_paired = set()  # Track files that have already been paired
    
    # Check if there are multiple files for the same wordID
    for file_path in required_files:
        mic = file_path.split('\\')[-1].split('_')[-1].split('.')[0]
        
        # Ensure that the microphone exists in the pair_key_value (to avoid pairing issues)
        if mic in pair_key_value:
            # Make sure we're not pairing the same controlled file more than once
            controlled_file = pair_key_value[mic]
            if (file_path, controlled_file) not in already_paired:
                row = (file_path, controlled_file, word, personID[2], mic)
                rows.append(row)
                already_paired.add((file_path, controlled_file))  # Mark this pair as used

    return rows


def process_and_write_rows(dataset, word_list, intelligibility, pair_key, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dysarthria', 'controlled', 'word', 'intelligibility', 'microphone'])
        
        already_written = set()  # Track which rows have been written to avoid duplication
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for personID in intelligibility.itertuples(index=False):
                # Select the appropriate gender key based on the speaker (M or F)
                key = pair_key['male'] if personID[0][0] == "M" else pair_key['female']
                
                for word, wordID in word_list.itertuples(index=False):
                    futures.append(executor.submit(process_row, personID, word, wordID, dataset, key[wordID]))

            for future in tqdm(futures, desc="Processing rows", total=len(futures)):
                rows = future.result()
                for row in rows:
                    # Only write rows that haven't been written before
                    if (row[0], row[1], row[4]) not in already_written:
                        writer.writerow(row)
                        already_written.add((row[0], row[1], row[4]))  # Track written rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get the path of the input (old) and output folders from the user."
    )
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the old directory')
    parser.add_argument('-t', '--type', type=str, required=True, help='Type of data to use')
    parser.add_argument('-pm', '--pairmale', type=str, required=True, help='Male ID to be paired')
    parser.add_argument('-pf', '--pairfemale', type=str, required=True, help='Female ID to be paired')
    parser.add_argument('-pk', '--pairkey', type=str, help='Female ID to be paired')
    args = parser.parse_args()
    root = args.directory
    data_type = args.type
    pair_male_id = args.pairmale
    pair_female_id = args.pairfemale
    pair_key_file = args.pairkey

    dataset = os.path.join(root, data_type)
    
    word_list, intelligibility = get_metadata(root, data_type)
    if pair_key_file:
        with open(pair_key_file,'r') as fl:
            pair_key = json.load(fl)
    else:
        pair_key = generate_pair_key(word_list,dataset,pair_male_id,pair_female_id)
        with open("pair_key.json","w") as fl:
            json.dump(pair_key,fl)
    
    process_and_write_rows(dataset,word_list,intelligibility,pair_key,os.path.join(root,"annotation.csv"))

    # for personID in intelligibility.itertuples(index=False):
    #     for word, wordID in word_list.itertuples(index=False):
    #         required = os.path.join(dataset, "dysarthria", personID[0], f'{personID[0]}*{wordID}*')
    #         required = glob.glob(required)
    #         for files in required:
    #             mic = files.split('\\')[-1].split('_')[-1].split('.')[0]
    #             row = files,pair_key[wordID][mic],word,personID[2],mic

if __name__ == '__main__':
    main()
