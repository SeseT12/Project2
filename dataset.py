from torch.utils.data import Dataset
import cv2
import config
import os
import xml.etree.ElementTree as ET
import torch

def get_image_paths(image_folder):
    image_paths = []
    for path in sorted(os.listdir(image_folder)):
        image_paths.append(os.path.join(image_folder, path))
        
    return image_paths

def preprocess_input(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, output = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return output

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for bbox in root.iter('bbox'):
        bboxes.append([int(bbox[0].text), int(bbox[1].text), int(bbox[2].text), int(bbox[3].text)])

    return bboxes


class CaptchaDataset(Dataset):
    def __init__(self, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = get_image_paths(config.INPUT_PATH)
        self.target_paths = get_image_paths(config.TARGET_PATH)
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        # load the image from disk
        # and read the associated mask from disk in grayscale mode
        image = preprocess_input(image_path)#cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = cv2.imread(self.target_paths[idx], cv2.IMREAD_UNCHANGED)
        target = cv2.resize(target, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            target = self.transforms(target)
        # return a tuple of the image and its mask
        return (image, target)

class CaptchaDataset2(Dataset):
    def __init__(self, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.image_paths = get_image_paths(config.INPUT_PATH)
        self.target_paths = get_image_paths(config.TARGET_PATH)
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        # load the image from disk
        # and read the associated mask from disk in grayscale mode
        image = preprocess_input(image_path)#cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

        bboxes = parse_xml(self.target_paths[idx])
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd



        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            #target = self.transforms(target)
        # return a tuple of the image and its mask
        return (image, target)