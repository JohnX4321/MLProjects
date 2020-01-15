import numpy as np
import argparse
import imutils,time,cv2

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to  model")
ap.add_argument("-c","--classes",required=True,help="class labels")
ap.add_argument("-i","--image",required=True,help="path i/p image")
ap.add_argument("-l","--colors",type=str,help="color labels")
ap.add_argument("-w","--width",type=int,default=500,help="desired width")
args=vars(ap.parse_args())

CLASSES=open(args["classes"]).read().strip().split("\n")

if args["colors"]:
    COLORS=open(args["colors"]).read().strip().split("\n")
    COLORS=[np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS=np.array(COLORS,dtype="uint8")

else:
    np.random.seed(42)
    COLORS=np.random.randint(0,255,size=(len(CLASSES)-1,3),dtype="uint8")
    COLORS=np.vstack([[0,0,0],COLORS]).astype("uint8")

legend=np.zeros(((len(CLASSES)*25)+25,300,3),dtype="uint8")

for (i,(className,color)) in enumerate(zip(CLASSES,COLORS)):
    color=[int(c) for c in color]
    cv2.putText(legend,className,(5,(i*25)+17),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.rectangle(legend,(100,(i*25)),(300,(i*25)+25),tuple(color),-1)


print("loading model")
net=cv2.dnn.readNet(args["model"])

image=cv2.imread(args["image"])
image=imutils.resize(image,width=args["width"])
blob=cv2.dnn.blobFromImage(image,1/255.0,(1024,512),0,swapRB=True,crop=False)

net.setInput(blob)
start=time.time()
output=net.forward()
end=time.time()

print("inference took {:.4f} sec".format(end-start))

(numClasses,height,width)=output.shape[1:4]

classMap=np.argmax(output[0],axis=0)
mask=COLORS[classMap]

mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

# perform a weighted combination of the input image with the mask to
# form an output visualization
output = ((0.4 * image) + (0.6 * mask)).astype("uint8")

# show the input and output images
cv2.imshow("Legend", legend)
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)

