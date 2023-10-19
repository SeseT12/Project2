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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='Directory containing training images', type=str)
    parser.add_argument('--xml-dir', help='Directory containing XML files', type=str)
    parser.add_argument('--output-model-name', help='Name for the trained model', type=str, default="model")
    parser.add_argument('--epochs', help='Number of training epochs', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.image_dir is None:
        print("Please specify the directory containing the training images")
        exit(1)

    if args.xml_dir is None:
        print("Please specify the directory containing the XML files")
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

    images = []
    labels = []

    for image_file in os.listdir(args.image_dir):
        if image_file.endswith(".png"):
            captcha_text = os.path.splitext(image_file)[0]
            xml_file = os.path.join(args.xml_dir, captcha_text + '.xml')
            bboxes = parse_xml(xml_file)
            raw_data = io.imread(os.path.join(args.image_dir, image_file))
            rgb_data = color.convert_colorspace(raw_data, 'RGB', 'RGB')
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                char_img = rgb_data[y_min:y_max, x_min:x_max]
                resized_char_img = np.array(Image.fromarray(char_img).resize((32, 32)))
                images.append(resized_char_img)
                label = captcha_text[i]
                label_index = captcha_symbols.find(label)
                label_array = np.zeros(num_classes)
                label_array[label_index] = 1
                labels.append(label_array)

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    input_shape = X_train[0].shape

    model = create_model(input_shape, num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=True)
    ]

    try:
        model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, validation_data=(X_test, y_test), callbacks=callbacks)
    except KeyboardInterrupt:
        print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name + '_resume.h5')
        model.save_weights(args.output_model_name + '_resume.h5')

    model_json = model.to_json()
    with open(args.output_model_name + '.json', 'w') as json_file:
        json_file.write(model_json)

if __name__ == '__main__':
    main()
