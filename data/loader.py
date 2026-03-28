# This file is responsible to load and save all the data.
from raw_data.amazon import GetAmazonData
from raw_data.imdb import GetImdbData
from raw_data.sentiment import GetSentimentData
from raw_data.tripadvisor import GetHotelData  

class RawDataLoader:
    def __init__(self, total_size: int = 15500):  # CHANGE 2: renamed subset_size to total_size and defaulted to 15,500
        self.amazon = GetAmazonData()
        self.imdb = GetImdbData()
        self.sentiment = GetSentimentData()
        self.hotels = GetHotelData()
        self.total_size = total_size  # CHANGE 3: renamed attribute to total_size

    def saveData(self) -> None:
        # CHANGE 4: removed the accepts_subset flag entirely — all datasets
        # now use the same total_size and produce identical splits
        # (train: 10,000 | val: 2,000 | test: 3,000 | ads: 500)
        datasets = [
            ("Amazon product review", self.amazon),
            ("IMDb movie review",     self.imdb),
            ("Sentiment140 review",   self.sentiment),
            ("Hotels review",         self.hotels),  # CHANGE 5: updated display name to match new dataset
        ]

        for name, data_class in datasets:  # CHANGE 6: simplified loop — no more accepts_subset unpacking
            print(f"Loading and saving the {name} data")
            try:
                data_class.getData(total_size=self.total_size)  # CHANGE 7: all classes now called with total_size uniformly
                print("Loading the data complete")
                print("Saving the data....")
                data_class.saveData()
                print(f"Successfully saved {name} data!\n")
            except Exception as e:
                print(f"Error processing {name} data: {e}\n")


if __name__ == '__main__':
    raw_data_loader = RawDataLoader(total_size=15500)  # CHANGE 8: updated instantiation to use total_size
    raw_data_loader.saveData()