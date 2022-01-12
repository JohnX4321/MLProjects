import torch,os

DATASET_PATH=os.path.join("dataset","train")
IMAGE_DATASET_PATH=os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH=os.path.join(DATASET_PATH,"masks")
TEST_SPLIT=0.15
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY=True if DEVICE=="cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    def __init__(self,imagePaths,maskPaths,transforms):
        self.imagePaths=imagePaths
        self.maskPaths=maskPaths
        self.transforms=transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath=self.imagePaths[idx]
        image=cv2.imread(imagePath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.maskPaths[idx],0)
        if self.transforms is not None:
            image=self.transforms(image)
            mask=self.transforms(mask)
        return (image,mask)

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
    def __init__(self,inChannels,outChannels):
        super.__init__()
        self.conv1=Conv2d(inChannels,outChannels,3)
        self.relu=ReLU()
        self.conv2=Conv2d(outChannels,outChannels,3)

    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self,channels=(3,16,32,64)):
        super.__init__()
        self.encBlocks=ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])
        self.pool=MaxPool2d(2)

    def forward(self,x):
        blockOutputs=[]
        for block in self.encBlocks:
            x=block(x)
            blockOutputs.append(x)
            x=self.pool(x)
        return blockOutputs

class Decoder(Module):
        def __init__(self, channels=(64, 32, 16)):
            super().__init__()
            # initialize the number of channels, upsampler blocks, and
            # decoder blocks
            self.channels = channels
            self.upconvs = ModuleList(
                [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                 for i in range(len(channels) - 1)])
            self.dec_blocks = ModuleList(
                [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])

        def forward(self, x, encFeatures):
            # loop through the number of channels
            for i in range(len(self.channels) - 1):
                # pass the inputs through the upsampler blocks
                x = self.upconvs[i](x)
                # crop the current features from the encoder blocks,
                # concatenate them with the current upsampled features,
                # and pass the concatenated output through the current
                # decoder block
                encFeat = self.crop(encFeatures[i], x)
                x = torch.cat([x, encFeat], dim=1)
                x = self.dec_blocks[i](x)
            # return the final decoder output
            return x

        def crop(self, encFeatures, x):
            # grab the dimensions of the inputs, and crop the encoder
            # features to match the dimensions
            (_, _, H, W) = x.shape
            encFeatures = CenterCrop([H, W])(encFeatures)
            # return the cropped features
            return encFeatures

class UNet(Module):
    def __init__(self,encChannels=(3,16,32,64),decChannels=(64,32,16),nbClasses=1,retainDim=True,outSize=(INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)):
            super().__init__()
            self.encoder=Encoder(encChannels)
            self.decoder=Decoder(decChannels)
            self.head=Conv2d(decChannels[-1],nbClasses,1)
            self.retainDim=retainDim
            self.outSize=outSize

    def forward(self,x):
        encFeatures=self.encoder(x)
        decFeatures=self.decoder(encFeatures[::-1][0],encFeatures[::-1][1:])
        map=self.head(decFeatures)
        if self.retainDim:
            map=F.interpolate(map,self.outSize)
        return map


from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
	test_size=TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,
		INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(DEVICE)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // BATCH_SIZE
testSteps = len(testDS) // BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)
# serialize the model to disk
torch.save(unet, MODEL_PATH)


import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()


def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,
			INPUT_IMAGE_HEIGHT))
    # make the channel axis to be the leading one, add a batch
    # dimension, create a PyTorch tensor, and flash it to the
    # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)
    # make the prediction, pass the results through the sigmoid
    # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
    # filter out the weak predictions and convert them to integers
        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
    # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask)
