from torch.utils.data import Dataset
import cv2
import config
import os

def get_image_paths(image_folder):
    image_paths = []
    for path in os.listdir(image_folder):
        image_paths.append(os.path.join(image_folder, path))
        
    return image_paths

def preprocess_input(image_path):
    image = cv2.imread("output/Input/!*mS.png")#cv2.imread(image_path)
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, output = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('image', output)
    cv2.waitKey(0)
    print("Preprocess")


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
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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

preprocess_input("")
