import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from skimage import io, color
import string
import random
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import xml.etree.ElementTree as ET
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

def create_model(num_classes):
    model = models.Sequential()
    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def binarize_image(image):
    gray_image = rgb2gray(image)
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh
    return binary_image.astype(int)

def preprocess_image(image):
    resized_image = tf.image.resize(image, (32, 32))
    binarized_image = binarize_image(resized_image)
    return resized_image

def preprocess_label(label, captcha_symbols):
    label = np.array([captcha_symbols.index(ch) for ch in label])
    return label

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, input_dir, target_dir, batch_size, captcha_symbols):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.captcha_symbols = captcha_symbols
        self.files = [file.split('.')[0] for file in os.listdir(input_dir)]
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for file in batch_files:
            input_image_path = os.path.join(self.input_dir, file + '.png')
            target_xml_path = os.path.join(self.target_dir, file + '.xml')

            raw_data = io.imread(input_image_path)
            processed_data = preprocess_image(raw_data)
            batch_x.append(processed_data)

            bboxes = parse_xml(target_xml_path)
            label = file.split('_')[0]
            batch_y.append(preprocess_label(label, self.captcha_symbols))

        return np.array(batch_x), np.array(batch_y)


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for bbox in root.iter('bbox'):
        x_min = int(bbox.find('x_min').text)
        y_min = int(bbox.find('y_min').text)
        x_max = int(bbox.find('x_max').text)
        y_max = int(bbox.find('y_max').text)
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--batch-size', help='Batch size for training', type=int, default=32)
    parser.add_argument('--epochs', help='Number of epochs for training', type=int, default=10)
    parser.add_argument('--train-input-dir', help='Path to the training input directory', type=str)
    parser.add_argument('--train-target-dir', help='Path to the training target directory', type=str)
    parser.add_argument('--validation-input-dir', help='Path to the validation input directory', type=str)
    parser.add_argument('--validation-target-dir', help='Path to the validation target directory', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    training_data = ImageSequence(args.train_input_dir, args.train_target_dir, args.batch_size, captcha_symbols)
    validation_data = ImageSequence(args.validation_input_dir, args.validation_target_dir, args.batch_size, captcha_symbols)

    num_classes = len(captcha_symbols)
    model = create_model(num_classes)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_data, epochs=args.epochs, validation_data=validation_data)

if __name__ == '__main__':
    main()
