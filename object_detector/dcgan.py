from torch.nn import *
from torch import flatten,nn


class DCGenerator(nn.Module):
    def __init__(self,inputDim=100,outputChannels=1):
        super(DCGenerator,self).__init__()
        #CONV->RELU->BN
        self.ct1=ConvTranspose2d(in_channels=inputDim,out_channels=128,kernel_size=4,stride=2,padding=0,bias=False)
        self.relu1=ReLU()
        self.batchNorm1=BatchNorm2d(128)

        #CONV->RELU->BN
        self.ct2=ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
        self.relu2=ReLU()
        self.batchNorm2=BatchNorm2d(64)

        #CONV->RELU->BN
        self.ct3=ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
        self.relu3=ReLU()
        self.batchNorm3=BatchNorm2d(32)

        self.ct4=ConvTranspose2d(in_channels=32,out_channels=outputChannels,kernel_size=4,stride=2,padding=1,bias=False)
        self.tanh=Tanh()

    def forward(self,x):
        x=self.ct1(x)
        x=self.relu1(x)
        x=self.batchNorm1(x)
        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)
        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchNorm3(x)
        x=self.ct4(x)
        output=self.tanh(x)
        return output

class Discriminator(nn.Module):
    def __init__(self,depth,alpha=0.2):
        super(Discriminator,self).__init__()
        self.conv1=Conv2d(in_channels=depth,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.leakyRelu1=LeakyReLU(alpha,inplace=True)
        self.conv2=Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.leakyRelu2=LeakyReLU(alpha,inplace=True)
        self.fc1=Linear(in_features=3136,out_features=512)
        self.leakyRelu3=LeakyReLU(alpha,inplace=True)
        self.fc2=Linear(in_features=512,out_features=1)
        self.sigmoid=Sigmoid()

    def forward(self,x):
        # pass the input through first set of CONV => RELU layers
        x = self.conv1(x)
        x = self.leakyRelu1(x)
        # pass the output from the previous layer through our second
        # set of CONV => RELU layers
        x = self.conv2(x)
        x = self.leakyRelu2(x)
        # flatten the output from the previous layer and pass it
        # through our first (and only) set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu3(x)
        # pass the output from the previous layer through our sigmoid
        # layer outputting a single value
        x = self.fc2(x)
        output = self.sigmoid(x)
        # return the output
        return output

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import argparse,torch,cv2,os
import numpy as np

def weights_init(model):
    classname=model.__class__.__name__
    if classname.find("Conv")!=-1:
        nn.init.normal_(model.weight.data,0.0,0.02)
    elif classname.find("BatchNorm")!=-1:
        nn.init.normal_(model.weight.data,1.0,0.02)
        nn.init.constant_(model.bias.data,0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=20,
	help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128,
	help="batch size for training")
args = vars(ap.parse_args())
# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataTransforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
trainData=MNIST(root="data",train=True,download=True,transform=dataTransforms)
testData=MNIST(root="data",train=False,download=True,transform=dataTransforms)
data=torch.utils.data.ConcatDataset((trainData,testData))
dataLoader=DataLoader(data,shuffle=True,batch_size=BATCH_SIZE)

steps=len(dataLoader.dataset)//BATCH_SIZE
gen=DCGenerator(inputDim=100,outputChannels=1)
gen.apply(weights_init)
gen.to(DEVICE)
disc=Discriminator(depth=1)
disc.apply(weights_init)
disc.to(DEVICE)
genOpt=Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.999),weight_decay=0.0002/NUM_EPOCHS)
discOpt=Adam(disc.parameters(),lr=0.0002,betas=(0.5,0.999),weight_decay=0.0002/NUM_EPOCHS)
criteria=BCELoss()

benchmarkNoise=torch.rand(256,100,1,1,device=DEVICE)
real=1
fake=1
for e in range(NUM_EPOCHS):
    epochLossG=0
    epochLossD=0
    for x in dataLoader:
        # zero out the discriminator gradients
        disc.zero_grad()
        # grab the images and send them to the device
        images = x[0]
        images = images.to(DEVICE)
        # get the batch size and create a labels tensor
        bs = images.size(0)
        labels = torch.full((bs,), real, dtype=torch.float,
                            device=DEVICE)
        # forward pass through discriminator
        output = disc(images).view(-1)
        # calculate the loss on all-real batch
        errorReal = criteria(output, labels)
        # calculate gradients by performing a backward pass
        errorReal.backward()
        # randomly generate noise for the generator to predict on
        noise = torch.randn(bs, 100, 1, 1, device=DEVICE)
        # generate a fake image batch using the generator
        fake = gen(noise)
        labels.fill_(fake)
        # perform a forward pass through discriminator using fake
        # batch data
        output = disc(fake.detach()).view(-1)
        errorFake = criteria(output, labels)
        # calculate gradients by performing a backward pass
        errorFake.backward()
        # compute the error for discriminator and update it
        errorD = errorReal + errorFake
        discOpt.step()
        # set all generator gradients to zero
        gen.zero_grad()
        # update the labels as fake labels are real for the generator
        # and perform a forward pass  of fake data batch through the
        # discriminator
        labels.fill_(real)
        output = disc(fake).view(-1)
        # calculate generator's loss based on output from
        # discriminator and calculate gradients for generator
        errorG = criteria(output, labels)
        errorG.backward()
        # update the generator
        genOpt.step()
        # add the current iteration loss of discriminator and
        # generator
        epochLossD += errorD
        epochLossG+=errorG
        if (e + 1) % 2 == 0:
            # set the generator in evaluation phase, make predictions on
            # the benchmark noise, scale it back to the range [0, 255],
            # and generate the montage
            gen.eval()
            images = gen(benchmarkNoise)
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]
            # build the output path and write the visualization to disk
            p = os.path.join(args["output"], "epoch_{}.png".format(
                str(e + 1).zfill(4)))
            cv2.imwrite(p, vis)
            # set the generator to training mode
            gen.train()
