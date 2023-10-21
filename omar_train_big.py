#!/usr/bin/env python3

# Import the necessary libraries
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

class ImageSequence(keras.utils.Sequence):
    def __init__(self, image_directory_name, xml_directory_name, text_directory_name, captcha_length, batch_size, captcha_symbols):
        self.image_directory_name = image_directory_name
        self.xml_directory_name = xml_directory_name
        self.text_directory_name = text_directory_name
        self.captcha_length = captcha_length
        self.batch_size = batch_size
        self.captcha_symbols = captcha_symbols
        self.num_classes = len(captcha_symbols)

        file_list = os.listdir(self.image_directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list)) # abcd -> abcd.png
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size * self.captcha_length, 32, 32, 3), dtype=np.float32)
        y = np.zeros((self.batch_size * self.captcha_length, self.num_classes), dtype=np.uint8)


        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            if(len(self.files.keys()) == 1):
                break
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            #raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            #rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            # raw_data = io.imread(os.path.join(self.directory_name, random_image_file))
            # rgb_data = color.convert_colorspace(raw_data, 'RGB', 'BGR')
            # processed_data = numpy.array(rgb_data) / 255.0
            # X[i] = processed_data

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.

            image_number = os.path.splitext(random_image_file)[0]
            xml_file = os.path.join(self.xml_directory_name, image_number + '.xml')
            bboxes = parse_xml(xml_file)
            with open(os.path.join(self.text_directory_name, image_number + '.txt'), 'r') as text_file:
                captcha_text = text_file.read().strip()
            raw_data = io.imread(os.path.join(self.image_directory_name, random_image_file))
            rgb_data = color.convert_colorspace(raw_data, 'RGB', 'RGB')
            for j, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, x_min)  # Check and adjust x_min
                y_min = max(0, y_min)  # Check and adjust y_min
                x_max = min(rgb_data.shape[1], x_max)  # Check and adjust x_max
                y_max = min(rgb_data.shape[0], y_max)  # Check and adjust y_max
                char_img = rgb_data[y_min:y_max, x_min:x_max]
                if char_img.size == 0:
                    continue  # Skip if the bounding box results in an empty image
                resized_char_img = np.array(Image.fromarray(char_img).resize((32, 32)))
                X[i + j] = resized_char_img
                label = captcha_text[j]
                label_index = self.captcha_symbols.find(label)
                label_array = np.zeros(self.num_classes)
                label_array[label_index] = 1
                y[i + j] = label_array

            # # Understand the following
            # random_image_label = random_image_label.split('_')[0]
            #
            # for j, ch in enumerate(random_image_label):
            #     y[j][i, :] = 0
            #     y[j][i, self.captcha_symbols.find(ch)] = 1
        return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-image-dir', help='Directory containing training images', type=str)
    parser.add_argument('--train-xml-dir', help='Directory containing XML files', type=str)
    parser.add_argument('--train-text-dir', help='Directory containing text files', type=str)
    parser.add_argument('--validate-image-dir', help='Directory containing training images', type=str)
    parser.add_argument('--validate-xml-dir', help='Directory containing XML files', type=str)
    parser.add_argument('--validate-text-dir', help='Directory containing text files', type=str)
    parser.add_argument('--captcha-length', help='Number of characters in captchas used for training and validation', type=int)
    parser.add_argument('--output-model-name', help='Name for the trained model', type=str)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--epochs', help='Number of training epochs', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.train_image_dir is None:
        print("Please specify the directory containing the training images")
        exit(1)

    if args.train_xml_dir is None:
        print("Please specify the directory containing the XML files")
        exit(1)

    if args.train_text_dir is None:
        print("Please specify the directory containing the text files")
        exit(1)

    if args.validate_image_dir is None:
        print("Please specify the directory containing the training images")
        exit(1)

    if args.validate_xml_dir is None:
        print("Please specify the directory containing the XML files")
        exit(1)
    if args.validate_text_dir is None:
        print("Please specify the directory containing the text files")
        exit(1)

    if args.captcha_length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.batch_size is None:
        print("Please specify the training batch size")
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

    input_shape = (32, 32, 3)

    training_data = ImageSequence(args.train_image_dir, args.train_xml_dir, args.train_text_dir, args.captcha_length, args.batch_size, captcha_symbols)
    validation_data = ImageSequence(args.train_image_dir, args.train_xml_dir, args.train_text_dir, args.captcha_length, args.batch_size, captcha_symbols)

    # with tf.device('/device:GPU:0'):
    with tf.device('/device:CPU:0'):
    # with tf.device('/device:XLA_CPU:0'):
        model = create_model(input_shape, num_classes)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(1e-3, amsgrad=True), metrics=['accuracy'])
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=3),
            keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=True)
        ]

        try:
            model.fit_generator(generator=training_data, epochs=args.epochs, validation_data=validation_data, callbacks=callbacks, use_multiprocessing=True)
            # model.fit(X_train, y_train, batch_size=32, epochs=args.epochs, validation_data=(X_test, y_test), callbacks=callbacks)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name + '_resume.h5')
            model.save_weights(args.output_model_name + '_resume.h5')

        model_json = model.to_json()
        with open(args.output_model_name + '.json', 'w') as json_file:
            json_file.write(model_json)

if __name__ == '__main__':
    main()
