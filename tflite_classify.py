import csv
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import argparse
import math
import tflite_runtime.interpreter as tflite
# python3 classify.py --captchas omar_captchas --symbols symbols.txt --model_tflite final.tflite --user_id alammu --output output.csv
def classify():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captchas', help='Path to Captchas Folder', type=str)
    parser.add_argument('--symbols', help='Path to Symbols.txt', type=str)
    parser.add_argument('--model_tflite', help='Path to Classification Model TFLite', type=str)
    parser.add_argument('--user_id', help='User ID to generate the CSV for', type=str)
    parser.add_argument('--output', help='Output File Path', type=str)
    args = parser.parse_args()

    symbols_file = open(args.symbols)
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    interpreter = tflite.Interpreter(model_path=args.model_tflite)
    interpreter.allocate_tensors()

    with open(args.output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.user_id])

        for image_path in sorted(os.listdir(args.captchas)):
            file_path = os.path.join(args.captchas, image_path)
            captcha_text = classify_captcha(file_path, interpreter, captcha_symbols, image_path)
            print(captcha_text)
            data = [file_path.replace(args.captchas + "/", ""), captcha_text]
            writer.writerow(data)

def classify_captcha(captcha_path, interpreter, symbols, image_path):
    xml_path = os.path.join('omar_xml', image_path.replace(".png", ".xml"))
    bboxes = parse_xml(xml_path)
    captcha_text = ""
    for bbox in bboxes:
        captcha_text = captcha_text + classify_character(bbox, captcha_path, interpreter, symbols)

    return captcha_text

def classify_character(bbox, image_path, interpreter, symbols):
    data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    char_img = data[int(np.floor(y_min)):int(np.ceil(y_max)), int(np.floor(x_min)):int(np.ceil(x_max))]
    resized_char_img = np.array(cv2.resize(char_img, (32, 32)))
    reshaped_img = np.expand_dims(resized_char_img, axis=0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], reshaped_img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = decode(symbols, prediction)
    return predicted_label

def decode(characters, prediction):
    decoded_symbol = ""
    max_index = np.argmax(prediction)
    decoded_symbol = characters[max_index]
    return decoded_symbol

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for bbox in root.findall('bbox'):
        x_min = int(math.floor(float(bbox.find('x_min').text)))
        y_min = int(math.floor(float(bbox.find('y_min').text)))
        x_max = int(math.ceil(float(bbox.find('x_max').text)))
        y_max = int(math.ceil(float(bbox.find('y_max').text)))
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

if __name__ == "__main__":
    classify()
