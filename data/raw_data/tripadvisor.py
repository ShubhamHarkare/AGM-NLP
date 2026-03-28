#VIBE-CODE Alert: This file is written using the Claude Sonnet 4.6
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

class GetHotelData:
    '''
    This class is responsible for gathering Booking.com hotel review data,
    selecting a random subset, splitting it, and storing it locally in a
    structured directory.

    The Booking.com dataset has a unique structure where each row contains
    BOTH a positive and negative review text as separate columns. We extract
    each column as a separate labeled example:
        - negative_review column → label 0
        - positive_review column → label 1

    Split sizes (out of 15,500 total):
        - Train:    10,000
        - Val:       2,000
        - Test:      3,000
        - ADS Pool:    500
    '''

    def __init__(self):
        self.dataset_name = os.environ.get('HOTEL_DATA', 'enelpol/booking_com_reviews')
        self.output_dir = "data/hotel"

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ads_data = None

        # CHANGE 1: minimum token threshold to filter out empty or trivial reviews
        self.min_words = 10

    def _preprocess(self, raw_data) -> Dataset:
        '''
        CHANGE 2: added dedicated preprocessing method to handle the
        Booking.com specific column structure.

        Steps:
            1. Extract negative reviews → label 0
            2. Extract positive reviews → label 1
            3. Filter out rows where text is empty or below min_words threshold
            4. Combine both into a single dataset with {text, label, domain} schema
        '''

        df = raw_data.to_pandas()

        # CHANGE 3: extract negative reviews as label 0
        # FIX: column names are capitalized — Negative_Review not negative_review
        negative_df = pd.DataFrame({
            'text':   df['Negative_Review'],
            'label':  0,
            'domain': 'hotel'
        })

        # CHANGE 4: extract positive reviews as label 1
        # FIX: column names are capitalized — Positive_Review not positive_review
        positive_df = pd.DataFrame({
            'text':   df['Positive_Review'],
            'label':  1,
            'domain': 'hotel'
        })

        # CHANGE 5: combine both into one dataframe
        combined_df = pd.concat([negative_df, positive_df], ignore_index=True)

        # CHANGE 6: filter out empty texts and texts below the minimum word threshold
        # this handles rows where guests left one side of the review blank
        combined_df = combined_df[
            combined_df['text'].notna() &                              # remove nulls
            (combined_df['text'].str.strip() != '') &                  # remove empty strings
            (combined_df['text'].str.split().str.len() >= self.min_words)  # remove very short texts
        ]

        # CHANGE 7: reset index after filtering so there are no gaps
        combined_df = combined_df.reset_index(drop=True)

        # FIX: explicitly keep only the 3 columns we need — guarantees clean
        # {text, label, domain} schema and prevents pandas __index_level_0__
        # or any other stray column from sneaking into the saved dataset
        combined_df = combined_df[['text', 'label', 'domain']]

        print(f"Total rows after preprocessing: {len(combined_df)}")
        print(f"Positive reviews: {len(combined_df[combined_df['label'] == 1])}")
        print(f"Negative reviews: {len(combined_df[combined_df['label'] == 0])}")
        print(f"Columns: {list(combined_df.columns)}")  # should print: ['text', 'label', 'domain']

        # CHANGE 8: convert back to HuggingFace Dataset — preserve_index=False
        # prevents the pandas index from becoming an extra column
        return Dataset.from_pandas(combined_df, preserve_index=False)

    def getData(self, total_size=15500):
        # CHANGE 9: load only the train split since that's where the bulk
        # of the data lives — we do our own splitting downstream
        raw_data = load_dataset(self.dataset_name, split='train')

        # CHANGE 10: preprocess BEFORE shuffling and selecting so that
        # label assignment is based on actual content, not row position
        full_data = self._preprocess(raw_data)

        # Shuffle and select total_size rows with fixed seed for reproducibility
        full_data = full_data.shuffle(seed=42).select(range(total_size))

        # Stage 1 — carve out ADS pool (500) from the full 15,500
        ads_split = full_data.train_test_split(test_size=500, seed=42)
        self.ads_data = ads_split['test']           # 500 rows
        remaining = ads_split['train']              # 15,000 rows remaining

        # Stage 2 — carve out test set (3,000) from remaining 15,000
        test_split = remaining.train_test_split(test_size=3000, seed=42)
        self.test_data = test_split['test']         # 3,000 rows
        remaining = test_split['train']             # 12,000 rows remaining

        # Stage 3 — carve out val set (2,000) from remaining 12,000
        val_split = remaining.train_test_split(test_size=2000, seed=42)
        self.val_data = val_split['test']           # 2,000 rows
        self.train_data = val_split['train']        # 10,000 rows

    def saveData(self):
        if any(split is None for split in [self.train_data, self.val_data, self.test_data, self.ads_data]):
            print("No data found. Please run getData() first.")
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

        print(f"Train rows:    {self.train_data.num_rows}")   # expected: 10,000
        print(f"Val rows:      {self.val_data.num_rows}")     # expected:  2,000
        print(f"Test rows:     {self.test_data.num_rows}")    # expected:  3,000
        print(f"ADS pool rows: {self.ads_data.num_rows}")     # expected:    500

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
    loader = GetHotelData()
    loader.getData()   # loads, preprocesses, and splits 15,500 rows
    loader.saveData()  # saves train / val / test / ads to disk