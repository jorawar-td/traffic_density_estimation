import cv2
from main2 import *
from datetime import datetime
from datetime import timedelta

class cut():
    def __init__(self,file_path,line,graph_path,time):
        self.file_path = file_path
        self.line = line
        self.graph_path = graph_path
        self.time = time


#  Method to split the given video into multiple videos each having the time of 10 minutes
    def convert(self, vidPath,time):    
        # vidPath = 'du.mp4'
        datetime_object = datetime.strptime(time, '%I:%M%p')
        x = 0
        cap = cv2.VideoCapture(vidPath)
        frameNumbers = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        f1 = 0
        var = fps * 60 * 10
        print("var",var)
        f2 = var
        segRange = []
        if frameNumbers // f2 > 0:
            while(f2+var<frameNumbers):
                segRange.append((f1,f2))
                f1=f2
                f2=f2+var
            else:
                f1=f2
                f2 = frameNumbers - f2
                segRange.append((f1,f2))
        else:
            f1 = 0
            f2 = frameNumbers
            segRange.append((f1,f2))
        print("SegRangeList ",segRange)

        filename = 'test_image\image'
        # segRange = [(0,150)] # a list of starting/ending frame indices pairs
        print("segRange",segRange)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        for idx,(begFidx,endFidx) in enumerate(segRange):
            new_path = filename+str(idx)+".mp4"
            writer = cv2.VideoWriter(new_path,fourcc,fps,size)
            x += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
            ret = True # has frame returned
            while(cap.isOpened() and ret and writer.isOpened()):
                ret, frame = cap.read()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if frame_number < endFidx:
                    writer.write(frame)
                else:
                    break
            writer.release()
            print("Video Created")
            var = Count(new_path,self.line,self.graph_path)
            di = var.vehicle()

            b = datetime_object + timedelta(minutes = 10)
            t = str(b.time())+","
            p = str(di['person'])+","
            c = str(di['car'])+","
            b = str(di['bike'])+","
            o = str(di['other'])
            f = open("count.csv", "a+")
            f.write("\n")
            f.write(t)
            f.write(p)
            f.write(c)
            f.write(b)
        self.graph(self.graph_path)
        return di

# Method to generate graph at the end
    def graph(self,path):
        data =pd.read_csv('count.csv')
        time = data['time']
        person = data['person']
        car = data['car']
        bike = data['bike']
        others = data['others']
        plt.plot(time,car,'o-',color="g",label="Test l1")
        plt.plot(time,bike,'o-',color="r", label="Train l1")
        plt.plot(time,person,'o-',color="b", label="Train l1")
        plt.plot(time,others,'o-',color="y", label="Train l1")

        plt.grid()
        plt.xlabel("Time(24 hrs)")
        plt.ylabel("Vehicles")
        fig = plt.savefig(path)
        return fig