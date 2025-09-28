from datasets import Dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm


def main():
    subfolders = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
    base_data_path = Path("/home/christian/vae/data_nb_1")
    output_path = Path("/home/christian/vae/data_nb_1_combined")
    input_folders = [base_data_path / subfolder for subfolder in subfolders]

    datasets = []
    for folder in tqdm(input_folders, desc="Loading datasets"):
        dataset = Dataset.load_from_disk(folder)
        datasets.append(dataset)

    combined_dataset = concatenate_datasets(datasets)
    print(f"Combined dataset has {len(combined_dataset)} total samples")
    output_path.mkdir(parents=True, exist_ok=True)
    combined_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()