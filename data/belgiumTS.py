# Reference - https://www.datacamp.com/community/tutorials/tensorflow-tutorial#gs.5nbceCA
import warnings
warnings.filterwarnings("ignore")
import os
from skimage import io
from skimage import transform
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet():
    def __init__(self, images, labels):
        """Construct a dataset"""
        np.random.seed(1234)
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # # Reshape images to [num_samples, rows*columns]
        # if reshape:
        #     assert images.shape[3] == 1
        #     images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            ind0 = np.arange(self._num_examples)
            np.random.shuffle(ind0)
            self._images = self.images[ind0]
            self._labels = self.labels[ind0]

        # Next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            batch_rest_examples = self._num_examples - start
            batch_rest_images = self._images[start:self._num_examples]
            batch_rest_labels = self._labels[start:self._num_examples]

            if shuffle:
                ind = np.arange(self._num_examples)
                np.random.shuffle(ind)
                self._images = self.images[ind]
                self._labels = self.labels[ind]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - batch_rest_examples
            end = self._index_in_epoch
            batch_new_images = self._images[start:end]
            batch_new_labels = self._labels[start:end]
            return np.concatenate((batch_rest_images, batch_new_images), axis=0), np.concatenate(
                (batch_rest_labels, batch_new_labels), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


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


def read_datasets_helper(data_directory):
    '''Returns a list of images'''
    images, labels = load_data(data_directory)
    return images, labels


def resize_imgs(imgs_list, resize_shape):
    transformed_imgs = [transform.resize(img, resize_shape) for img in imgs_list]
    return np.array(transformed_imgs)


def read_datasets(data_dir="./data/", one_hot=True, rgb_gray=True, resize_pix=28, validation_size=0.1):
    train_data_dir = data_dir + "Training"
    test_data_dir = data_dir + "Testing"

    # Load training data
    train_imgs, train_labels = read_datasets_helper(train_data_dir)
    test_imgs, test_labels = read_datasets_helper(test_data_dir)

    # Resize images
    train_imgs = resize_imgs(train_imgs, (resize_pix, resize_pix))
    test_imgs = resize_imgs(test_imgs, (resize_pix, resize_pix))

    # Convert to grayscale from RGB
    if rgb_gray:
        train_imgs = rgb2gray(train_imgs)
        test_imgs = rgb2gray(test_imgs)
        # Reshape images to [num_samples, rows*columns]
        train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1] * train_imgs.shape[2])
        test_imgs = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1] * test_imgs.shape[2])

    # One-hot encode image labels
    if one_hot:
        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        test_labels = lb.transform(test_labels)

    if not 0 <= validation_size <= 1:
        raise ValueError(
            'Validation size should be between 0 and 1. Received: {}.'
                .format(validation_size))

    # Stratified shuffle split to creat validation images
    new_train_imgs, val_imgs, new_train_labels, val_labels = train_test_split(train_imgs, train_labels,
                                                                              test_size=validation_size,
                                                                              random_state=1234, stratify=train_labels)

    train = DataSet(new_train_imgs, new_train_labels)
    test = DataSet(test_imgs, test_labels)
    validation = DataSet(val_imgs, val_labels)

    return Datasets(train=train, test=test, validation=validation)

# @TODO: Add **options
def load_BelgiumTS(**kwargs):
    return read_datasets(**kwargs)
