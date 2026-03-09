from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv
import os

load_dotenv()


class GetAmazonData:
    '''
    This class is responsible for gathering Amazon Polarity data, selecting
    a random subset, splitting it, and storing it locally in a structured
    directory.

    IMPORTANT — Amazon Polarity actual schema:
        Native columns: 'label', 'text', 'label_text'
        - 'text' already exists natively — no combining needed
        - 'label' is already binary {0, 1} — no remapping needed
        - 'label_text' is dropped — not needed downstream

    Split sizes (out of 15,500 total):
        - Train:    10,000
        - Val:       2,000
        - Test:      3,000
        - ADS Pool:    500
    '''

    def __init__(self):
        self.dataset_name = os.environ.get('AMAZON_DATA', 'fancyzhx/amazon_polarity')
        self.output_dir = "data/amazon"

        self.train_data = None
        self.val_data   = None
        self.test_data  = None
        self.ads_data   = None

    def _preprocess(self, dataset):
        '''
        Amazon Polarity actual schema: ['label', 'text', 'label_text']
        - 'text' already exists natively — no combining needed
        - 'label' is already binary {0, 1} — no remapping needed
        - just add 'domain' and drop 'label_text'
        '''
        # FIX 1: text and label already exist — just add domain column
        dataset = dataset.map(lambda x: {'domain': 'amazon'})

        # FIX 2: drop label_text — keep only {text, label, domain}
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names
             if col not in ('text', 'label', 'domain')]
        )

        print(f"Columns after preprocessing: {dataset.column_names}")  # should be ['text', 'label', 'domain']
        return dataset

    def getData(self, total_size=15500):
        # FIX 3: renamed hf_train/hf_test to avoid variable name collision
        # with our own test_split variable used in the splitting logic below
        hf_train  = load_dataset(self.dataset_name, split='train')
        hf_test   = load_dataset(self.dataset_name, split='test')
        full_data = concatenate_datasets([hf_train, hf_test])

        # Preprocess before shuffling so column changes apply to full dataset
        full_data = self._preprocess(full_data)

        # Shuffle with fixed seed for reproducibility
        full_data = full_data.shuffle(seed=42).select(range(total_size))

        # Stage 1 — carve out ADS pool (500) from the full 15,500
        ads_split     = full_data.train_test_split(test_size=500, seed=42)
        self.ads_data = ads_split['test']    # 500 rows
        remaining     = ads_split['train']   # 15,000 rows remaining

        # Stage 2 — carve out test set (3,000) from remaining 15,000
        test_split     = remaining.train_test_split(test_size=3000, seed=42)
        self.test_data = test_split['test']  # 3,000 rows
        remaining      = test_split['train'] # 12,000 rows remaining

        # Stage 3 — carve out val set (2,000) from remaining 12,000
        val_split       = remaining.train_test_split(test_size=2000, seed=42)
        self.val_data   = val_split['test']  # 2,000 rows
        self.train_data = val_split['train'] # 10,000 rows

    def saveData(self):
        if any(split is None for split in [self.train_data, self.val_data, self.test_data, self.ads_data]):
            print("No data found. Please run getData() first.")
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

        print(f"Train rows:    {self.train_data.num_rows}")  # expected: 10,000
        print(f"Val rows:      {self.val_data.num_rows}")    # expected:  2,000
        print(f"Test rows:     {self.test_data.num_rows}")   # expected:  3,000
        print(f"ADS pool rows: {self.ads_data.num_rows}")    # expected:    500

        splits = {
            "train": self.train_data,
            "val":   self.val_data,
            "test":  self.test_data,
            "ads":   self.ads_data,
        }

        for split_name, split_data in splits.items():
            split_path = os.path.join(self.output_dir, split_name)
            split_data.save_to_disk(split_path)
            print(f"Saved {split_name} split to {split_path}")


# Example Usage:
if __name__ == "__main__":
    loader = GetAmazonData()
    loader.getData()   # loads, preprocesses, and splits 15,500 rows
    loader.saveData()  # saves train / val / test / ads to disk