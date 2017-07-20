# Reference - https://www.datacamp.com/community/tutorials/tensorflow-tutorial#gs.5nbceCA
import os
from skimage import io


class load_data():
    @staticmethod
    def load_data(data_directory):
        directories = [d for d in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, d))]
        labels = []
        images = []
        for d in directories:
            label_dir = os.path.join(data_directory, d)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir)
                          if f.endswith(".ppm")]
            for f in file_names:
                images.append(io.imread(f))
                labels.append(int(d))
        return images, labels

    def training_data(self, data_directory="./data/Training"):
        train_images, train_labels = self.load_data(data_directory)
        return train_images, train_labels

    def testing_data(self, data_directory="./data/Testing"):
        test_images, test_labels = self.load_data(data_directory)
        return test_images, test_labels
