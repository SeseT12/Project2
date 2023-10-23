#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from skimage import io, color
import xml.etree.ElementTree as ET
from PIL import Image

def create_model(input_shape, num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding='same'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120, activation='tanh'))
    model.add(keras.layers.Dense(84, activation='tanh'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for bbox in root.findall('bbox'):
        x_min = int(bbox.find('x_min').text)
        y_min = int(bbox.find('y_min').text)
        x_max = int(bbox.find('x_max').text)
        y_max = int(bbox.find('y_max').text)
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

class CaptchaDataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, xml_dir, text_dir, symbols, batch_size=32, validation=False):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.text_dir = text_dir
        self.symbols = symbols
        self.batch_size = batch_size
        self.captcha_symbols = self._load_symbols()
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        if validation:
            self.image_files = self.image_files[:int(len(self.image_files) * 0.2)]
        else:
            self.image_files = self.image_files[int(len(self.image_files) * 0.2):]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_image_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        labels = []

        for image_file in batch_image_files:
            image_number = os.path.splitext(image_file)[0]
            xml_file = os.path.join(self.xml_dir, image_number + '.xml')
            bboxes = parse_xml(xml_file)
            with open(os.path.join(self.text_dir, image_number + '.txt'), 'r') as text_file:
                captcha_text = text_file.read().strip()
            raw_data = io.imread(os.path.join(self.image_dir, image_file))
            rgb_data = color.convert_colorspace(raw_data, 'RGB', 'RGB')
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(rgb_data.shape[1], x_max)
                y_max = min(rgb_data.shape[0], y_max)
                char_img = rgb_data[y_min:y_max, x_min:x_max]
                if char_img.size == 0:
                    continue
                resized_char_img = np.array(Image.fromarray(char_img).resize((32, 32)))
                images.append(resized_char_img)
                label = captcha_text[i]
                label_index = self.captcha_symbols.find(label)
                label_array = np.zeros(len(self.captcha_symbols))
                label_array[label_index] = 1
                labels.append(label_array)

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        np.random.shuffle(self.image_files)

    def _load_symbols(self):
        with open(self.symbols) as symbols_file:
            captcha_symbols = symbols_file.readline().strip()
        return captcha_symbols


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for bbox in root.findall('bbox'):
        x_min = int(bbox.find('x_min').text)
        y_min = int(bbox.find('y_min').text)
        x_max = int(bbox.find('x_max').text)
        y_max = int(bbox.find('y_max').text)
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='Directory containing training images', type=str)
    parser.add_argument('--xml-dir', help='Directory containing XML files', type=str)
    parser.add_argument('--text-dir', help='Directory containing text files', type=str)
    parser.add_argument('--output-model-name', help='Name for the trained model', type=str)
    parser.add_argument('--epochs', help='Number of training epochs', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.image_dir is None:
        print("Please specify the directory containing the training images")
        exit(1)

    if args.xml_dir is None:
        print("Please specify the directory containing the XML files")
        exit(1)

    if args.text_dir is None:
        print("Please specify the directory containing the text files")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    num_classes = len(captcha_symbols)

    train_data_generator = CaptchaDataGenerator(args.image_dir, args.xml_dir, args.text_dir, args.symbols, validation=False)
    validation_data_generator = CaptchaDataGenerator(args.image_dir, args.xml_dir, args.text_dir, args.symbols, validation=True)

    input_shape = (32, 32, 3)

    model = create_model(input_shape, num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(1e-5, amsgrad=True), metrics=['accuracy'])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=50),
        keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=True)
    ]

    try:
        model.fit(train_data_generator, epochs=args.epochs, validation_data=validation_data_generator, callbacks=callbacks)
    except KeyboardInterrupt:
        print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name + '_resume.h5')
        model.save_weights(args.output_model_name + '_resume.h5')

    model_json = model.to_json()
    with open(args.output_model_name + '.json', 'w') as json_file:
        json_file.write(model_json)

if __name__ == '__main__':
    main()
