#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
from captcha import ImageCaptcha
import xml.etree.ElementTree as ET
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_generator = ImageCaptcha(width=args.width, height=args.height, fonts=["The Jjester.otf"])

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "Input")):
        print("Creating input directory " + os.path.join(args.output_dir, "Input"))
        os.makedirs(os.path.join(args.output_dir, "Input"))

    if not os.path.exists(os.path.join(args.output_dir, "Target")):
        print("Creating target directory " + os.path.join(args.output_dir, "Target"))
        os.makedirs(os.path.join(args.output_dir, "Target"))

    if not os.path.exists(os.path.join(args.output_dir, "Captchas")):
        print("Creating target directory " + os.path.join(args.output_dir, "Captchas"))
        os.makedirs(os.path.join(args.output_dir, "Captchas"))

    for i in range(args.count):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        input_image_path = os.path.join(os.path.join(args.output_dir, "Input"), str(i)+'.png')
        target_xml_path = os.path.join(os.path.join(args.output_dir, "Target"), str(i) + '.xml')
        captcha_path = os.path.join(os.path.join(args.output_dir, "Captchas"), str(i) + '.txt')

        """""""""
        if os.path.exists(input_image_path):
            version = 1
            while os.path.exists(os.path.join(os.path.join(args.output_dir, "Input"), random_str + '_' + str(version) + '.png')):
                version += 1
            input_image_path = os.path.join(os.path.join(args.output_dir, "Input"), random_str + '_' + str(version) + '.png')
            target_xml_path = os.path.join(os.path.join(args.output_dir, "Target"), random_str + '_' + str(version) + '.xml')
        """""""""

        input_image, bboxes = captcha_generator.generate_image_with_bounding_box(random_str)
        cv2.imwrite(input_image_path, numpy.array(input_image))
        bboxes_to_xml(bboxes, target_xml_path)
        text_file = open(captcha_path, "w+")
        text_file.write(random_str)
        text_file.close()

        #cv2.imwrite(target_image_path, numpy.array(target_image))


def bboxes_to_xml(bboxes, file_path):
    data = ET.Element('bboxes')
    for bbox in bboxes:
        bbox_element = ET.SubElement(data, 'bbox')
        x_min = ET.SubElement(bbox_element, 'x_min')
        x_min.text = str(bbox[0])
        y_min = ET.SubElement(bbox_element, 'y_min')
        y_min.text = str(bbox[1])
        x_max = ET.SubElement(bbox_element, 'x_max')
        x_max.text = str(bbox[2])
        y_max = ET.SubElement(bbox_element, 'y_max')
        y_max.text = str(bbox[3])

        b_xml = ET.tostring(data)

    with open(file_path, "wb") as f:
        f.write(b_xml)

from PIL.Image import new as createImage, Image, QUAD, BILINEAR
from PIL.ImageDraw import Draw, ImageDraw
from PIL.ImageFont import FreeTypeFont, truetype
if __name__ == '__main__':
    main()
    """""""""
    captcha_generator = ImageCaptcha(width=120, height=120, fonts=["The Jjester.otf"])
    image = createImage('RGB', (120, 120), (255,0,100))
    draw = Draw(image)
    input_image, bboxes = captcha_generator.generate_image_with_bounding_box('abcde')
    bboxes_to_xml(bboxes)
    
    tree = ET.parse('test.xml')
    root = tree.getroot()
    bboxes = []
    for bbox in root.iter('bbox'):
        print(bbox)
        bboxes.append([int(bbox[0].text), int(bbox[1].text), int(bbox[2].text), int(bbox[3].text)])

    image_with_bb = createImage('RGB', (120, 120), (0, 0, 0))
    for bbox2 in bboxes:
        print(bbox2)
        Draw(image_with_bb).rectangle(bbox2, outline='Yellow')

    input_image.show()
    image_with_bb.show()
    """""""""
