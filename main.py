import cv2

class analyze_image:
    def __init__(self,imagename,outputfile):
        self.image = None
        self.imagename = imagename
        self.outputfile = outputfile
        self.classnames = None

    #reads data needed for program.
    def readdata(self):
        self.image = cv2.imread(self.imagename)
        with open('coco.names','rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

    def showimage(self):
        cv2.imshow("Output",self.image)
        cv2.waitKey(0)

    def outputImg(self):
        cv2.imwrite(self.outputfile,self.image)

    def createnet(self):
        net = cv2.dnn_DetectionModel('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt','frozen_inference_graph.pb')
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        return net
    
    def analyze(self):
        self.readdata()
        net = self.createnet()
        classIds, confs, bbox = net.detect(self.image,confThreshold=0.45)
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(self.image,box,color=(0,255,0),thickness=2)
            cv2.putText(self.image,self.classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            print(self.classNames[classId-1])
        self.outputImg()

a = analyze_image("mrc.jpg","ouput.png")
a.analyze()




    
