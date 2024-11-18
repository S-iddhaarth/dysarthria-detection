from collections import defaultdict
from tqdm import tqdm

def get_length_distribution(dataset):
    length_distribution_per_label = defaultdict(lambda: defaultdict(int))
    all_lengths = defaultdict(int)
    for i in tqdm(dataset,total=len(dataset),desc="collecting data"):
        length = i['audio'].shape[0]
        length_distribution_per_label[i['label']][length] += 1
        all_lengths[length] += 1
    return all_lengths,length_distribution_per_label

from concurrent.futures import ThreadPoolExecutor, as_completed


def process_item(item):
    length = item['audio'].shape[0]
    label = item['label']
    return length, label

def get_length_distribution_multi_thread(dataset, max_workers=4):
    length_distribution_per_label = defaultdict(lambda: defaultdict(int))
    all_lengths = defaultdict(int)
    length_distribution_per_label_linear = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in dataset}

        for future in tqdm(as_completed(future_to_item), total=len(dataset), desc="collecting data"):
            try:
                length, label = future.result()
                length_distribution_per_label[label][length] += 1
                length_distribution_per_label_linear[label].append(length)
                all_lengths[length] += 1
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return all_lengths, length_distribution_per_label,length_distribution_per_label_linear

# Example usage
# dataset = ... (your dataset)
# all_lengths, length_distribution_per_label = get_length_distribution(dataset)
