import math

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
from PIL import Image
import numpy as np
from datetime import datetime

import config
import unet
import dataset
import dice_loss


class ModelTrainer:

    def train(self):
        # define transformations
        transformations = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                                 config.INPUT_IMAGE_WIDTH)),
                                              transforms.ToTensor()])
        ds = dataset.CaptchaDataset(transformations)
        print(len(ds))
        proportions = [.85, .15]
        lengths = [int(p * len(ds)) for p in proportions]
        lengths[-1] = len(ds) - sum(lengths[:-1])
        train_ds, test_ds = torch.utils.data.random_split(ds, lengths)
        print(f"[INFO] found {len(train_ds)} examples in the training set...")
        print(f"[INFO] found {len(test_ds)} examples in the test set...")
        # create the training and test data loaders
        train_loader = DataLoader(train_ds, shuffle=True,
                                  batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                                  num_workers=os.cpu_count())
        test_loader = DataLoader(test_ds, shuffle=False,
                                 batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                                 num_workers=os.cpu_count())

        # initialize our UNet model
        #unet = u_net.UNet().to(config.DEVICE)

        model = unet.UNet().to(config.DEVICE)
        # initialize loss function and optimizer
        loss_function = dice_loss.DiceLoss()#DiceBCELoss()
        optimizer = Adam(model.parameters(), lr=config.INIT_LR)
        #optimizer = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9, weight_decay=math.exp(-4))
        # calculate steps per epoch for training and test set
        train_steps = len(train_ds) // config.BATCH_SIZE
        test_steps = len(test_ds) // config.BATCH_SIZE
        # initialize a dictionary to store training history
        H = {"train_loss": [], "test_loss": []}

        # loop over epochs
        print("[INFO] training the network...")
        start_time = time.time()
        for e in tqdm(range(config.NUM_EPOCHS)):
            # set the model in training mode
            model.train()
            # initialize the total training and validation loss
            total_train_loss = 0
            total_test_loss = 0
            loss_list = []
            # loop over the training set
            for (i, (x, y)) in enumerate(train_loader):
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # perform a forward pass and calculate the training loss
                prediction = model(x)
                loss = loss_function(prediction, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # add the loss to the total training loss so far
                total_train_loss += loss.detach()
                loss_list.append(loss.detach().cpu().numpy())
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                for (x, y) in test_loader:
                    # send the input to the device
                    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                    # make the predictions and calculate the validation loss
                    prediction = model(x)
                    loss = loss_function(prediction, y)
                    total_test_loss += loss
                    loss_list.append(loss.detach().cpu().numpy())

            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / train_steps
            avg_test_loss = total_test_loss / test_steps
            loss_std = np.std(loss_list)
            # update our training history
            H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            H["test_loss"].append(avg_test_loss.cpu().detach().numpy())
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
            print("Train loss: {:.6f}, Test loss: {:.4f}, std: {:.4f}".format(
                avg_train_loss, avg_test_loss, loss_std))
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - start_time))

        # serialize the model to disk
        torch.save(model, os.path.join(config.BASE_OUTPUT, model_name + ".pth"))


if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.train()