from urllib.request import urlretrieve
import zipfile
import argparse
import time

# train_url = "http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip" #(173 MB)
# test_url = "http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip" #(76.5MB)

extract_path = "./data/"


def train_data(train_url):
    zip_fldr = "./data/BelgiumTSC_Training.zip"
    print("Downloading Training Data...")
    urlretrieve(train_url, zip_fldr)
    zip_ref = zipfile.ZipFile(zip_fldr)
    print("\n Extracting Folder Contents...")
    zip_ref.extractall(extract_path)
    zip_ref.close()


def test_data(test_url):
    zip_fldr = "./data/BelgiumTSC_Testing.zip"
    print("Downloading Testing Data...")
    urlretrieve(test_url, zip_fldr)
    zip_ref = zipfile.ZipFile(zip_fldr)
    print("\n Extracting Folder Contents...")
    zip_ref.extractall(extract_path)
    zip_ref.close()


if __name__ == '__main__':
    '''Download data for Belgium Traffic Signs'''
    parser = argparse.ArgumentParser(
        description='Download belgium traffic sign data set for classification from given URL')
    parser.add_argument('-train_url', type=str, help="URL to download training dataset",
                        default="http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip")
    parser.add_argument('-test_url', type=str, help="URL to download testing dataset",
                        default="http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip")
    args = parser.parse_args()

    start = time.time()
    train_data(args.train_url)
    test_data(args.test_url)
    print("It took ", time.time() - start, 'seconds to download and extract Belgium TS Classification data.')
