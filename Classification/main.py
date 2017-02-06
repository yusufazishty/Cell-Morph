import Tkinter
#import tkMessageBox
import tkFileDialog
from PIL import ImageTk, Image
import numpy as np
#import matplotlib
import cv2
import skimage.io
import skimage.feature
from skimage import feature
import scipy
import os
#import time
#import png
from skimage.measure import label, regionprops
import os.path
import pickle
import copy
import sklearn
from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

sizeFrame = 200
state = 0
datasetPath = os.getcwd()+"\\Dataset" 
#datasetPath = 'F:\Work\Freelance\Bu Chastine\Dataset'
cacheAllFitur = 'AllFitur.p'
cachearrayFitur = 'ArrayFitur.p'
K=4 #untuk cross validation
 
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist
# NOTE
# Nucleus Grayscale value = 29
# Cytoplasma Grayscale value 76
# Background Graysclae value = 15

class simpleapp_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.iconbitmap(default='picture.ico')
        self.parent = parent
        self.initialize()

    def ImageFrame(self,frame, inputCol,inputRow,labelText = "Tes"):
        self.frame = self.mainFrame()
        self.frame.grid(column=inputCol,row=inputRow,sticky='EW',padx=5,pady=5)
        self.frame.label = Tkinter.Label(self.frame,text=labelText)
        self.frame.label.grid(column=0,row=0,sticky='EW',padx=5,pady=5)
        self.frame.frameImage = self.mainFrame(self.frame,'flat')
        self.frame.frameImage.grid(column=0,row=1,sticky='EW',padx=5,pady=5)
        return self.frame
        
    def mainFrame(self,frame = None,relief ='ridge'):
        if frame == None:
            self.frame = Tkinter.Frame(self)
        else :
            self.frame = Tkinter.Frame(frame)
        self.frame['height']=sizeFrame
        self.frame['width']=sizeFrame
        self.frame['borderwidth']=1
        self.frame['relief']=relief
        return self.frame

    def initialize(self):
        self.grid()

        self.InputFrame = self.mainFrame(None,'flat')
        self.InputFrame.grid(columnspan=3,row=0,sticky='EW',padx=5,pady=5)

        self.viewImage = self.mainFrame(None,'flat')
        self.viewImage.grid(columnspan=3,row=1,sticky='EW',padx=5,pady=5)

        self.fCitraAsli = self.ImageFrame(self.viewImage,0,1,'Citra Asli')
        self.fCitraSegmentasi = self.ImageFrame(self.viewImage,1,1,'Citra Segmentasi')
        self.frame3 = self.ImageFrame(self.viewImage,2,1,'2')
        self.frame4 = self.ImageFrame(self.viewImage,0,2,'3')

        self.pathCitraAsli = Tkinter.StringVar()
        self.pathCitraAsli.set('Dateset/light_dysplastic/148494967-148494986-001.BMP')
        self.label = Tkinter.Label(self.InputFrame,text="Citra Asli : ")
        self.label.grid(column=0,row=0,sticky='EW',padx=5,pady=5)
        self.entryFilepathAsli = Tkinter.Entry(self.InputFrame,width=75,state='readonly',textvariable=self.pathCitraAsli)
        self.entryFilepathAsli.grid(column=1,row=0,sticky='EW',padx=5,pady=5)
        self.button = Tkinter.Button(self.InputFrame, text="Select a File", command= lambda: self.openfile(self.fCitraAsli, self.entryFilepathAsli))
        self.button.grid(column=2,row=0,sticky='EW',padx=5,pady=5)

        self.pathCitraSegmentasi = Tkinter.StringVar()
        self.pathCitraSegmentasi.set('Dataset/light_dysplastic/148494967-148494986-001-d.BMP')
        self.label = Tkinter.Label(self.InputFrame,text="Citra Segmentasi : ")
        self.label.grid(column=0,row=1,sticky='EW',padx=5,pady=5)
        self.entryFilepathSegmentasi = Tkinter.Entry(self.InputFrame,width=75,state='readonly',textvariable=self.pathCitraSegmentasi)
        self.entryFilepathSegmentasi.grid(column=1,row=1,sticky='EW',padx=5,pady=5)
        self.button = Tkinter.Button(self.InputFrame, text="Select a File", command= lambda: self.openfile(self.fCitraSegmentasi, self.entryFilepathSegmentasi))
        self.button.grid(column=2,row=1,sticky='EW',padx=5,pady=5)

        self.button = Tkinter.Button(self.InputFrame, text="Proses", command = self.proses)
        self.button.grid(column=1,row=2,sticky='EW',padx=5,pady=5)

        self.proses()

    def openfile(self,frame,entryFilepath):
        inputFile = tkFileDialog.askopenfilename()
        image = Image.open(inputFile)
        image = image.resize((sizeFrame-20,sizeFrame-20), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.labelImage = Tkinter.Label(frame,image=photo)
        self.labelImage.image = photo
        self.labelImage.grid(column=0,row=1,sticky='EW',padx=5,pady=5)
        entryFilepath.configure(state='normal')
        entryFilepath.delete(0,'end')
        entryFilepath.insert(0,inputFile)
        entryFilepath.configure(state='readonly')

    def cropImagebyPixelValue(self,imageAsli,imageSegmentasi,height,width,pixel):
        minX = -1
        maxX = -1
        minY = -1
        maxY = -1
        for x in range(0,height):
            for y in range(0,width):
                if imageSegmentasi[x][y] == pixel:
                    if minX == -1:
                        minX = x
                        maxX = x
                    elif minX > x:
                        minX = x
                    elif maxX < x:
                        maxX = x

                    if minY == -1:
                        minY = y
                        maxY = y
                    elif minY > y:
                        minY = y
                    elif maxY < y:
                        maxY = y

        newHeight = maxX - minX
        newWidth = maxY - minY
        imgNew = np.zeros((newHeight,newWidth,3), np.uint8)
        for x in range(0,newHeight):
            for y in range(0,newWidth):
                if imageSegmentasi[x+minX][y+minY] == pixel:
                    imgNew[x][y] = imageAsli[x+minX][y+minY]
        return imgNew
        
    def segmentation(self):
        img = cv2.imread(inputFile)
        img = cv2.medianBlur(img,5)
        filename = 'Segmentation.png'
        write = cv2.imwrite(filename,img)
        image = Image.open(filename)
        image = image.resize((sizeFrame-20,sizeFrame-20), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.labelImage = Tkinter.Label(self.frame2,image=photo)
        self.labelImage.image = photo
        self.labelImage.grid(column=0,row=1,sticky='EW',padx=5,pady=5)
    

    def ekstraksiFitur(self,pixel,param):
        height, width = self.imgAsli.shape[:2]
        obj = self.cropImagebyPixelValue(self.imgAsli,self.imgSegmentasi,height,width,pixel)
        
        Bobj = obj[:,:,0]
        Gobj = obj[:,:,1]
        Robj = obj[:,:,2]

        newHeight, newWidth = obj.shape[:2]
        objGrayscale = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)

        ret,thresh = cv2.threshold(objGrayscale,0,255,0)

        label_img = label(thresh)
        region = skimage.measure.regionprops(label_img)

        self.Fitur = {}

        self.Fitur['area'] = 0
        self.Fitur['longest'] = 0
        self.Fitur['shortest'] = 0
        self.Fitur['perimeter'] = 0
        for prop in region:
            self.Fitur['area'] += prop.area
            self.Fitur['longest'] += prop.major_axis_length
            self.Fitur['shortest'] += prop.minor_axis_length
            self.Fitur['perimeter'] += prop.perimeter
        self.Fitur['elongation'] = self.Fitur['longest'] / self.Fitur['shortest']
        self.Fitur['circle'] = 3.14 * self.Fitur['longest'] * self.Fitur['longest']
        self.Fitur['roundness'] = self.Fitur['area']/self.Fitur['circle']
        if param == 'cytoplasma':
            self.Fitur['compactness'] = self.Fitur['perimeter'] * self.Fitur['perimeter'] / self.Fitur['area']

        LBPnucleus = LocalBinaryPatterns(5,5)
        FiturLBPnucleus = LBPnucleus.describe(image = objGrayscale)
        self.Fitur['lbp'] = FiturLBPnucleus

        arr = np.reshape(obj,newHeight*newWidth*3)
        array = list(arr)
        array = [x for x in array if x != 0]
        self.Fitur['brightness'] = np.mean(array)
        g = skimage.feature.greycomatrix(objGrayscale, [1], [0], levels=256, symmetric=False, normed=True)
        self.Fitur['homogeneity'] = skimage.feature.greycoprops(g,'homogeneity')[0][0]

        # MAXIMA NUCLEUS
        jum = 0
        footprint = np.ones((8,8), np.uint8)
        lm = scipy.ndimage.filters.maximum_filter(Bobj,footprint = footprint)
        msk = (Bobj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)

        lm = scipy.ndimage.filters.maximum_filter(Gobj,footprint = footprint)
        msk = (Gobj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)

        lm = scipy.ndimage.filters.maximum_filter(Robj,footprint = footprint)
        msk = (Robj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)
        self.Fitur['maxima'] = jum / 3

        # MINIMA NUCLEUS
        jum = 0
        footprint = np.ones((8,8), np.uint8)
        lm = scipy.ndimage.filters.minimum_filter(Bobj,footprint = footprint)
        msk = (Bobj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)

        lm = scipy.ndimage.filters.minimum_filter(Gobj,footprint = footprint)
        msk = (Gobj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)

        lm = scipy.ndimage.filters.minimum_filter(Robj,footprint = footprint)
        msk = (Robj == lm) #// convert local max values to binary mask
        (x,y) = msk.shape
        arr = np.reshape(msk,x*y)
        array = list(arr)
        array = [x for x in array if x != False]
        jum = jum + len(array)
        self.Fitur['minima'] = jum / 3
        return self.Fitur
    
    def proses(self):
        dumpedExist = os.path.isfile(datasetPath+'\\'+cacheAllFitur)
        if not dumpedExist:
            folderDataset = datasetPath;
            listdir = os.listdir(folderDataset)
            self.AllFitur = {}
            self.arrayFitur = []
            count = 0
            for num in listdir:
                listFile = os.listdir(folderDataset + '\\' + num)
                file = open(folderDataset + '\\' + num + '\\class.txt', 'r')
                text = file.read()
                text = text.split(' ')
                for i in range(0,len(listFile),2):
                    if listFile[i].endswith(".txt"):
                        continue
                    else:
                        array = []
                        data = {}
                        data['isNormal'] = text[0]
                        array.append(text[0])
                        data['class'] = text[1]
                        array.append(text[1])
                        Segmentasi =  listFile[i]
                        Ori =  listFile[i+1]
                        #print folderDataset+ '\\' + num + '\\' + Ori 
                        self.imgAsli = cv2.imread(folderDataset+ '\\' + num + '\\' + Ori , cv2.IMREAD_COLOR)
                        self.imgSegmentasi = cv2.imread(folderDataset+ '\\' + num + '\\' + Segmentasi, cv2.IMREAD_GRAYSCALE )
    
                        Fiturnucleus = self.ekstraksiFitur(29,'nucleus')
                        data['Nucleus'] = Fiturnucleus
                        for a in data['Nucleus']:
                            if a == 'lbp':
                                for b in data['Nucleus'][a]:
                                    array.append(b)
                            else:
                                array.append(data['Nucleus'][a])
                        Fiturcytoplasma = self.ekstraksiFitur(15,'cytoplasma')
                        data['Cytoplasma'] = Fiturcytoplasma
                        for a in data['Cytoplasma']:
                            if a=='lbp':
                                for b in data['Cytoplasma'][a]:
                                    array.append(b)
                            else:
                                array.append(data['Cytoplasma'][a])
                        self.arrayFitur.append(array)
                        self.AllFitur[count] = {}
                        self.AllFitur[count] = data
                        #print self.AllFitur[count]
                        count = count + 1
                        #print count
            self.arrayFitur = np.array(self.arrayFitur)
            #print self.arrayFitur[:,0]
            #bikin cache AllFitur
            pickle.dump(self.AllFitur,open(datasetPath+'\\'+cacheAllFitur,"wb"))
            pickle.dump(self.arrayFitur,open(datasetPath+'\\'+cachearrayFitur,"wb"))
            self.preClassifying()
        else:
            #tinggal baca cache AllFitur
            with open(datasetPath+'\\'+cacheAllFitur,"rb") as f:
                self.AllFitur = pickle.load(f)
            with open(datasetPath+'\\'+cachearrayFitur,"rb") as f:
                self.arrayFitur = pickle.load(f)
            self.preClassifying()
            
    def preClassifying(self):
        AllFitur = self.AllFitur
        arrayFitur = self.arrayFitur
        #arrayFitur=copy.deepcopy(cacheArrayFitur)
        #get distinctClass
        distinctClass = []
        for i in range(len(arrayFitur[:,1])):
            if arrayFitur[i,1] not in distinctClass:
                distinctClass.append(arrayFitur[i,1])
        #print distinctClass
        #make it to discrete
        discreteClass={}
        for i in range(len(distinctClass)):
            discreteClass[distinctClass[i]]=i
        #print discreteClass
        #transform to discrete class
        for i in range(len(arrayFitur[:,1])):
            arrayFitur[i][1]=discreteClass[arrayFitur[i][1]]
        tempInt = arrayFitur[:,0:2].astype(int)
        tempFloat = arrayFitur[:,2:].astype(float)
        #self.arrayFiturEdited = np.hstack([tempInt, tempFloat])
        arrayFiturEdited = np.hstack([tempInt, tempFloat])
        #normalize per column features
        for i in range(2,len(arrayFiturEdited[0])):
            #self.arrayFiturEdited[:,i]=preprocessing.normalize(self.arrayFiturEdited[:,i])
            arrayFiturEdited[:,i]=preprocessing.normalize(arrayFiturEdited[:,i])
           
        # prepare to pandas dataframe
        arrayFiturEditedPD = pd.DataFrame(arrayFiturEdited)
        twoClassPD = copy.deepcopy(arrayFiturEditedPD)
        multiClassPD = copy.deepcopy(arrayFiturEditedPD)
        
        # slicing for twoClassData
        del twoClassPD[1] #delete classs multi
        group = twoClassPD.groupby([0]).groups #index grouping
        for i in range(len(group)):
            if i==0:
                twoClass0 = twoClassPD.loc[group[i]] #kelas 0
                twoClass0 = twoClass0.as_matrix()
            else:
                twoClass1=twoClassPD.loc[group[i]] #kelas 1
                twoClass1=twoClass1.as_matrix()
        split0 = len(twoClass0)/K  
        split1 = len(twoClass1)/K
        splits0=[0]        
        for i in range(split0,len(twoClass0),split0):
            splits0.append(i)
        if splits0[-1]<len(twoClass0):
            splits0[-1]=len(twoClass0)
        splits0.append(-1000)
        
        splits1=[0]   
        for i in range(split1,len(twoClass1),split1):
            splits1.append(i)
        if splits1[-1]<len(twoClass1):
            splits1[-1]=len(twoClass1)
        splits1.append(-1000)
           
        featuresClass0=[[],[],[],[]]
        featuresClass1=[[],[],[],[]]
        #membagi 4 dari masing-masing klas
        #klas 0
        for i in range(len(splits0)):
            if splits0[i+1]==-1000:
                break;
            for j in range(splits0[i],splits0[i+1],1):
                if j==splits0[i]:
                    featuresClass0[i].append(twoClass0[j,:])
                else:
                    featuresClass0[i]=np.vstack([featuresClass0[i],twoClass0[j,:]])
        #klas 1
        for i in range(len(splits1)):
            if splits1[i+1]==-1000:
                break;
            for j in range(splits1[i],splits1[i+1],1):
                if j==splits1[i]:
                    featuresClass1[i].append(twoClass1[j,:])
                else:
                    featuresClass1[i]=np.vstack([featuresClass1[i],twoClass1[j,:]])
                
        #combine to crossval klas0 dan 1 untuk train-test
        self.dataTrainTestBiner=[]
        #dataTrainTestBiner=[]
        for i in range(K):
            tempTrainTest=[]
            a=[x for x in range(K)]
            tempTest=np.array([])
            tempTest=np.vstack([featuresClass0[i],featuresClass1[i]])
            tempTrain=np.array([])
            a.remove(i)
            tempTrain=np.array([])
            for j in range(len(a)):
                if j==0:
                    tempTrain=np.vstack([featuresClass0[a[j]],featuresClass1[a[j]]])
                else: 
                    sementara=np.vstack([featuresClass0[a[j]],featuresClass1[a[j]]])
                    tempTrain=np.vstack([tempTrain,sementara])
            tempTrainTest.append(tempTest)
            tempTrainTest.append(tempTrain)
            self.dataTrainTestBiner.append(tempTrainTest)
            #dataTrainTestBiner.append(tempTrainTest)
            
        # slicing for multiClassData
        del multiClassPD[0] #delete classs biner
        # grouping features based on the multi classes
        multiClassN=[[],[],[],[],[],[],[]] #this should be manual
        groupMulti = multiClassPD.groupby([1]).groups #index grouping
        for i in range(len(groupMulti)):
            tempClassN=multiClassPD.loc[groupMulti[i]] #kelas ke i
            tempClassN=tempClassN.as_matrix()
            multiClassN[i]=tempClassN
        #spliting every group to K partition
        split=[]
        for i in range(len(multiClassN)):
            banyakDataPerKelas = len(multiClassN[i])
            split.append(banyakDataPerKelas/K)        
        #splitting edge
        splits=[[0],[0],[0],[0],[0],[0],[0]]
        for i in range(len(multiClassN)):
            for j in range(split[i], len(multiClassN[i]), split[i]):
                #print j
                splits[i].append(j)
            if splits[i][-1]<len(multiClassN[i]):
                splits[i][-1]=len(multiClassN[i])
            splits[i].append(-1000)
        
           
        featuresClassMulti=[[],[],[],[],[],[],[]]
        #membagi 4 masing-masing kelas
        #di split dulu jadi 4K
        for i in range(len(multiClassN)):
            featuresClassEach=[[],[],[],[]]
            for j in range(len(splits[i])):
                if splits[i][j+1]==-1000:
                    break;
                for k in range(splits[i][j],splits[i][j+1],split[i]):
                    start = splits[i][j]
                    end = splits[i][j+1]
                    featuresClassEach[j]=multiClassN[i][start:end,:]
            featuresClassMulti[i]=featuresClassEach
               
        #combine to crossval train test data
        self.dataTrainTestMulti=[]
        #dataTrainTestMulti=[]
        for i in range(K):
            #print 'fold ke '+str(i)
            tempTrainTest=[]
            a=[x for x in range(K)]
            tempTest=np.array([])
            for j in range(0,len(featuresClassMulti),2):
                #print 'ambil data test'
                #print j
                if j==0:
                    tempTest=np.vstack([featuresClassMulti[j][i],featuresClassMulti[j+1][i]])
                elif j==6:
                    tempTest=np.vstack([tempTest, featuresClassMulti[j][i]])
                else:
                    tempTest=np.vstack([tempTest, featuresClassMulti[j][i]])
                    tempTest=np.vstack([tempTest, featuresClassMulti[j+1][i]])
            tempTrain=np.array([])
            a.remove(i)
            for k in range(len(a)):
                #print 'a[k] '+str(a[k])
                for j in range(0,len(featuresClassMulti),1):
                    #print 'ambil data train'
                    #print j
                    if j==0:
                        tempTrain = np.vstack([featuresClassMulti[j][a[k]]])
                    else:
                        tempTrain = np.vstack([tempTrain,featuresClassMulti[j][a[k]]])
                if k==0:
                    wrapper=np.vstack([tempTrain])
                else:
                    wrapper=np.vstack([wrapper,tempTrain])
            tempTrainTest.append(tempTest)
            tempTrainTest.append(wrapper)
            self.dataTrainTestMulti.append(tempTrainTest)
        self.classifying()
        
    def defineClassifier(self, method):
        if method=="knn":
            binerClassifier = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
            multiClassifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
        elif method=="svm":
            binerClassifier = svm.SVC()
            multiClassifier = svm.SVC()
        elif method=="bayes":
            binerClassifier= GaussianNB()
            multiClassifier= GaussianNB()
        elif method=="nn":
            binerClassifier= MLPClassifier(hidden_layer_sizes=(36, 2), random_state=1)
            #atau rule of thumb size=(36+2)*2/3 atau size=(36+7)*2/3
            multiClassifier= MLPClassifier(hidden_layer_sizes=(36, 7), random_state=1)
        return binerClassifier, multiClassifier
        
    def doClassify(self, method):
        print "classifying using "+method
        binerClassifier, multiClassifier = self.defineClassifier(method)
        dataTrainTestBiner = self.dataTrainTestBiner
        dataTrainTestMulti = self.dataTrainTestMulti
        tempResultBiner=[]
        tempResultMulti=[]
        if binerClassifier==multiClassifier==method:
            print "skip"
        else:
            for i in range(K):
                #yang biner dulu
                #bagian test
                test = dataTrainTestBiner[i][0]
                test_data = test[:,1:]; test_class = test[:,0] 
                #bagian train            
                train = dataTrainTestBiner[i][1]
                train_data = train[:,1:]; train_class = train[:,0]
                #laod classifier, lalu train
                classifier = binerClassifier
                classifier.fit(train_data,train_class)
                #uji data testing
                hasil_uji = classifier.predict(test_data)
                acc=self.evaluate(hasil_uji, test_class)
                tempResultBiner.append(acc)
                #resultBiner.append(acc)
                
                #yang multi
                #bagian test
                test = dataTrainTestMulti[i][0]
                test_data = test[:,1:]; test_class = test[:,0] 
                #bagian train
                train = dataTrainTestMulti[i][1]
                train_data = train[:,1:]; train_class = train[:,0]
                #laod classifier, lalu train
                classifier = multiClassifier
                classifier.fit(train_data,train_class)
                #uji data testing
                hasil_uji = classifier.predict(test_data)
                acc=self.evaluate(hasil_uji, test_class)
                tempResultMulti.append(acc)
                #resultMulti.append(acc)
                
        #last, tambah average 
        avgBiner=float(sum(tempResultBiner))/float(K)
        avgMulti=float(sum(tempResultMulti))/float(K)
        tempResultBiner.append(avgBiner)
        tempResultMulti.append(avgMulti)
        #tambahkan ke dictionary result
        self.resultBiner[method]=tempResultBiner
        self.resultMulti[method]=tempResultMulti
            
    def evaluate(self, hasil_uji, test_class):
        tp=0.0
        for j in range(len(test_class)):
            if test_class[j]==hasil_uji[j]:
                tp+=1
        acc=tp/float(len(test_class))
        return acc
        
    def classifying(self):
        self.resultBiner={}
        self.resultMulti={}
        methods=['knn', 'svm', 'bayes', 'nn']
        for a in range(len(methods)):
            method = methods[a]
            self.doClassify(method)
        print "result biner"
        print self.resultBiner
        print "result Multi"
        print self.resultMulti
            
if __name__ == "__main__":    
    app = simpleapp_tk(None)
    app.title('Cell Morph')
    app.mainloop()
