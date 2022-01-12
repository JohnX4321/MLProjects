import torch, os

BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])
BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
LABELS, BBOX = 1.0, 1.0

from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        image = image.permute(2, 0, 1)
        if self.transforms:
            image = self.transforms(image)
        return (image, label, bbox)


from torch.nn import *


class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses)
        )
        self.baseModel.fc = Identity()

    def forward(self, x):
        f = self.baseModel(x)
        bb = self.regressor(f)
        cl = self.classifier(f)
        return (bb, cl)


from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle, torch, time, cv2, os
from torch.utils.data import DataLoader
from torch.optim import Adam

data = []
labels = []
bboxes = []
imgPaths = []

for csvPath in paths.list_files(ANNOTS_PATH, validExts=(".csv")):
    rows = open(csvPath).read().strip().split("\n")
    for r in rows:
        r = r.split(",")
        (filename, startX, startY, endX, endY, label) = r
        imagePath = os.path.sep.join([IMAGES_PATH, label, filename])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        data.append(img)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imgPaths.append(imagePath)
data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imgPaths = np.array(imgPaths)
le = LabelEncoder()
labels = le.fit_transform(labels)
split = train_test_split(data, labels, bboxes, imgPaths, test_size=0.2, random_state=42)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

(trainImages, testImages) = torch.tensor(trainImages), torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels), torch.tensor(testLabels)
(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes), torch.tensor(testBBoxes)
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes), transforms=transforms)
testDS = CustomTensorDataset((testImages, testLabels, testBBoxes), transforms=transforms)
trainSteps = len(trainDS)
valSteps = len(testDS)
trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(),
                         pin_memory=PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
f = open(TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()
resnet = resnet50(pretrained=True)

for p in resnet.parameters():
    p.requires_grad = False
objectDet = ObjectDetector(resnet, len(le.classes_))
objectDet = objectDet.to(DEVICE)
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()
opt = Adam(objectDet.parameters(), lr=INIT_LR)
# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
     "val_class_acc": []}

for e in tqdm(range(NUM_EPOCHS)):
    objectDet.train()
    totalTrainLoss, totalValLoss = 0, 0
    trainCorrect, valCorrect = 0, 0
    for (i, l, b) in trainLoader:
        (i, l, b) = (i.to(DEVICE), l.to(DEVICE), b.to(DEVICE))
        predictions = objectDet(i)
        bboxLoss = bboxLossFunc(predictions[0], b)
        classLoss = classLossFunc(predictions[1], l)
        totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)
        opt.zero_grad()
        totalLoss.backward()
        opt.step()
        totalTrainLoss += totalLoss
        trainCorrect += (predictions[1].argmax(1) == l).type(torch.float).sum().item()
    with torch.no_grad():
        objectDet.eval()
        for (images, labels, bboxes) in testLoader:
            # send the input to the device
            (images, labels, bboxes) = (images.to(DEVICE),
                                        labels.to(DEVICE), bboxes.to(DEVICE))
            # make the predictions and calculate the validation loss
            predictions = objectDet(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (BBOX * bboxLoss) + \
                        (LABELS * classLoss)
            totalValLoss += totalLoss
            # calculate the number of correct predictions
            valCorrect += (predictions[1].argmax(1) == labels).type(
                torch.float).sum().item()
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDS)
    valCorrect = valCorrect / len(testDS)
    # update our training history
    H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_class_acc"].append(trainCorrect)
    H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_class_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
        avgValLoss, valCorrect))
# serialize the model to disk
print("[INFO] saving object detector model...")
torch.save(objectDet, MODEL_PATH)
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["total_val_loss"], label="total_val_loss")
plt.plot(H["train_class_acc"], label="train_class_acc")
plt.plot(H["val_class_acc"], label="val_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# save the training plot
plotPath = os.path.sep.join([PLOTS_PATH, "training.png"])
plt.savefig(plotPath)

from torchvision import transforms
import mimetypes
import argparse
import imutils
import pickle
import torch
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input image/text file of image paths")
args = vars(ap.parse_args())
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
    # load the image paths in our testing file
    imagePaths = open(args["input"]).read().strip().split("\n")
# load our object detector, set it evaluation mode, and label
# encoder from disk
print("[INFO] loading object detector...")
model = torch.load(MODEL_PATH).to(DEVICE)
model.eval()
le = pickle.loads(open(LE_PATH, "rb").read())
# define normalization transforms
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
    # load the image, copy it, swap its colors channels, resize it, and
    # bring its channel dimension forward
    image = cv2.imread(imagePath)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    # convert image to PyTorch tensor, normalize it, flash it to the
    # current device, and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms(image).to(DEVICE)
    image = image.unsqueeze(0)

    # predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = model(image)
    (startX, startY, endX, endY) = boxPreds[0]
    # determine the class label with the largest predicted
    # probability
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu()
    label = le.inverse_transform(i)[0]
    # resize the original image such that it fits on our screen, and
    # grab its dimensions
    orig = imutils.resize(orig, width=600)
    (h, w) = orig.shape[:2]
    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)
    cv2.rectangle(orig, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", orig)
    cv2.waitKey(0)
