from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()


class GetSentimentData:
    '''
    This class is responsible for gathering Sentiment140 (Twitter) data,
    selecting a random subset, splitting it, and storing it locally in a
    structured directory.

    IMPORTANT — Sentiment140 label schema:
        Native labels: 0 = negative, 4 = positive
        Remapped to:   0 = negative, 1 = positive
    Without this remapping the model would train on label 4 as positive
    which silently breaks training and evaluation.

    Split sizes (out of 15,500 total):
        - Train:    10,000
        - Val:       2,000
        - Test:      3,000
        - ADS Pool:    500
    '''

    def __init__(self):
        # FIX 1: added fallback default so code doesn't silently crash
        # if SENTIMENT_DATA is missing from the .env file
        self.dataset_name = os.environ.get('SENTIMENT_DATA', 'sentiment140')
        self.output_dir = "sentiment"

        self.train_data = None
        self.val_data   = None
        self.test_data  = None
        self.ads_data   = None

    def _preprocess(self, dataset):
        '''
        FIX 2: added preprocessing method to handle Sentiment140 specific issues:
            1. Remap labels from {0, 4} to {0, 1}
            2. Rename 'sentiment' column to 'label' for consistent schema
            3. Add domain column
        '''
        # FIX 3: remap label 4 → 1 so all datasets use binary {0, 1} labels
        # without this, positive examples have label=4 which breaks training
        dataset = dataset.map(lambda x: {
            'label':  0 if x['sentiment'] == 0 else 1,
            'domain': 'sentiment140'
        })

        # FIX 4: keep only the columns we need for a clean consistent schema
        # Sentiment140 has extra columns (date, user, query) we don't need
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names
             if col not in ('text', 'label', 'domain')]
        )

        return dataset

    def getData(self, total_size=15500):
        # FIX 5: Sentiment140 only has a 'train' split on HuggingFace
        # using split='all' would throw an error here
        full_data = load_dataset(self.dataset_name, split='train')

        # FIX 6: preprocess before shuffling so label remapping is applied
        # to the full dataset before any subset is selected
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
    loader = GetSentimentData()
    loader.getData()   # loads, preprocesses, and splits 15,500 rows
    loader.saveData()  # saves train / val / test / ads to disk