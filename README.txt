Here's how to run the various programs in this submission:

## Downloading the CAPTCHAs for the final submission

python3 download.py

For this file to run correctly, there needs to be a CSV called filenames.csv which lists all the CAPTCHA images in the 
same directory.

## Training the (segmented) character recognition model

python3 omar_train_big.py ----symbols symbols.txt --epochs 20 --image-dir ./training_data/Input --xml-dir 
./training_data/Target --text-dir ./training_data/Captchas --output-model-name model

In this example, all the training data is stored in the "training_data" directory. Here's a breakdown of the arguments 
passed and their corresponding subdirectories:

--image-dir: this is the directory that contains the CAPTCHA images for training
--xml-dir: this is the directory that contains the XML files with the character segmentation details for each character 
in the CAPTCHA
--text-dir: this is the directory that contains the targets for the CAPTCHAs, i.e. the characters the training data 
generator inserted in the image
