#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import xml.etree.ElementTree as ET

def decode(characters, y):
    decoded_symbols = []
    for prediction in y:
        max_index = numpy.argmax(prediction)
        decoded_symbols.append(characters[max_index])
    return ''.join(decoded_symbols)

def load_bboxes_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for bbox in root.iter('bbox'):
        x_min = int(bbox.find('x_min').text)
        y_min = int(bbox.find('y_min').text)
        x_max = int(bbox.find('x_max').text)
        y_max = int(bbox.find('y_max').text)
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--xml-dir', help='Where to read the XML files', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.xml_dir is None:
        print("Please specify the directory with XML files")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for filename in os.listdir(args.captcha_dir):
                if filename.endswith(".png"):
                    image_number = os.path.splitext(filename)[0]
                    xml_path = os.path.join(args.xml_dir, image_number + '.xml')
                    bboxes = load_bboxes_from_xml(xml_path)
                    raw_data = cv2.imread(os.path.join(args.captcha_dir, filename))
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                    for i, bbox in enumerate(bboxes):
                        x_min, y_min, x_max, y_max = bbox
                        char_img = rgb_data[y_min:y_max, x_min:x_max]
                        resized_char_img = numpy.array(cv2.resize(char_img, (32, 32)))
                        (c, h, w) = resized_char_img.shape
                        reshaped_img = resized_char_img.reshape([-1, c, h, w])
                        prediction = model.predict(reshaped_img)
                        predicted_label = decode(captcha_symbols, prediction)
                        output_file.write(image_number + ", " + predicted_label + "\n")

if __name__ == '__main__':
    main()
