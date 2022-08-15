import os
import csv
#import cv2
import pickle
import serial
import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

from mmWave import pc3
from turtle import width
from threading import Thread
from collections import deque
from sklearn.cluster import DBSCAN
from pyqtgraph.Qt import QtCore, QtGui

state_fall_lying = 0

# Data location
filepath = "./log/tester/"
if not os.path.isdir(filepath):
    os.makedirs(filepath)
data_number = str(len(os.listdir(filepath)))
path_data = filepath + data_number + ".csv"

# Write csv column name
with open(path_data, "a", newline="") as csvFile:
	writer = csv.writer(csvFile, dialect = "excel")
	writer.writerow(["time", "x", "y", "z", "state_code"])

# Import randomForestModel
with open('./model/randomForestModel2.pickle', 'rb') as f:
    randomForestModel = pickle.load(f)

z_fall_lying = []
smooth_zmean = []
coco = False
################### Class #######################################
class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
	def __init__(self, X, Y, Z, text):
		gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
		self.text = text
		self.X = X
		self.Y = Y
		self.Z = Z

	def setGLViewWidget(self, GLViewWidget):
		self.GLViewWidget = GLViewWidget

	def setText(self, text):
		self.text = text
		self.update()

	def setX(self, X):
		self.X = X
		self.update()

	def setY(self, Y):
		self.Y = Y
		self.update()

	def setZ(self, Z):
		self.Z = Z
		self.update()

	def paint(self):
		self.GLViewWidget.qglColor(QtCore.Qt.cyan)
		self.GLViewWidget.renderText(round(self.X), round(self.Y), round(self.Z), self.text)


class Custom3DAxis(gl.GLAxisItem):
	#Class defined to extend 'gl.GLAxisItem'
	def __init__(self, parent, color=(0,0,0,.6)):
		gl.GLAxisItem.__init__(self)
		self.parent = parent
		self.c = color
		
	def add_labels(self):
		#Adds axes labels. 
		x,y,z = self.size()
		#X label
		self.xLabel = CustomTextItem(X=x/2, Y=-y/20, Z=-z/20, text="X")
		self.xLabel.setGLViewWidget(self.parent)
		self.parent.addItem(self.xLabel)
		#Y label
		self.yLabel = CustomTextItem(X=-x/20, Y=y/2, Z=-z/20, text="Y")
		self.yLabel.setGLViewWidget(self.parent)
		self.parent.addItem(self.yLabel)
		#Z label
		self.zLabel = CustomTextItem(X=-x/20, Y=-y/20, Z=z/2, text="Z")
		self.zLabel.setGLViewWidget(self.parent)
		self.parent.addItem(self.zLabel)
		
	def add_tick_values(self, xticks=[], yticks=[], zticks=[]):
		#Adds ticks values. 
		x,y,z = self.size()
		xtpos = np.linspace(0, x, len(xticks))
		ytpos = np.linspace(0, y, len(yticks))
		ztpos = np.linspace(0, z, len(zticks))
		#X label
		for i, xt in enumerate(xticks):
			val = CustomTextItem(X=xtpos[i], Y=0, Z=0, text='{}'.format(xt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)
		#Y label
		for i, yt in enumerate(yticks):
			val = CustomTextItem(X=0, Y=ytpos[i], Z=0, text='{}'.format(yt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)
		#Z label
		for i, zt in enumerate(zticks):
			val = CustomTextItem(X=0, Y=0, Z=ztpos[i], text='{}'.format(zt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)


##########################################################################
def func(x, a, b):
    return a * np.exp(-b * x)
smooth_xr = []
smooth_xl = []
smooth_yr = []
smooth_yl = []
smooth_zr = []
smooth_zl = []

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()

####### camera position #######
w.setCameraPosition(distance=7, elevation=50, azimuth=90)

#size=50:50:50
g = gl.GLGridItem()
g.setSize(x=10,y=10,z=10)
#g.setSpacing(x=1, y=1, z=1, spacing=None)
w.addItem(g)

####### draw axis ######
axis = Custom3DAxis(w, color=(0.2,0.2,0.2,1.0))
axis.setSize(x=10, y=10, z=10)
xt = [0,5,10]  
axis.add_tick_values(xticks=xt, yticks=xt, zticks=xt)
w.addItem(axis)
w.setWindowTitle('Position Occupancy(Cluster)')

####### create box to represent device ######
verX = 0.0625
verY = 0.0
verZ = 0.125
zOffSet = 1.0
verts = np.empty((2,3,3))
verts[0,0,:] = [-verX, verY, verZ + 0.7]
verts[0,1,:] = [-verX, verY,-verZ + 0.7]
verts[0,2,:] = [verX,  verY,-verZ + 0.7]
verts[1,0,:] = [-verX, verY, verZ + 0.7]
verts[1,1,:] = [verX,  verY, verZ + 0.7]
verts[1,2,:] = [verX,  verY, -verZ + 0.7]

evmBox = gl.GLMeshItem(vertexes=verts,smooth=False,drawEdges=True,edgeColor=pg.glColor('g'),drawFaces=False)
w.addItem(evmBox)

# use USB-UART
#port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5)
port = serial.Serial("/dev/ttyTHS1", baudrate=921600, timeout=0.5)
line_stand = 0.7
line_lay = 0.35
line_lay_up = 0.5
radar = pc3.Pc3(port)

v6len = 0
v7len = 0
v8len = 0

pos = np.zeros((10000,3))
color = [1.0, 0.0, 0.0, 1.0]
# sp1 = gl.GLScatterPlotItem(pos=pos,color=color,size = 4.0)
# sp2 = gl.GLScatterPlotItem(pos=pos,color=color,size = 4.0)
line1 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line2 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line3 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line4 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line5 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line6 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line7 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line8 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line9 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line10 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line11 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
line12 = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]),color=[51,255,255, 255]) #自己+的
text_mdoe = QtGui.QTextItem

w.addItem(line1)
w.addItem(line2)
w.addItem(line3)
w.addItem(line4)
w.addItem(line5)
w.addItem(line6)
w.addItem(line7)
w.addItem(line8)
w.addItem(line9)
w.addItem(line10)
w.addItem(line11)
w.addItem(line12)
# w.addItem(sp1)
# w.addItem(sp2)
gcolorA = np.empty((10000,4), dtype=np.float32)
#generate a color opacity gradient

people_state = []
maxlabel = 1
def update():
	global gcolorA,sensorA,mapSum,box1, people_state, maxlabel

	# extract Labels
	# sp1.setData(pos=sensorA[:,[0,1,2]],color=gcolorA)
	# print("sensorA.type", sensorA.shape)
	# print("sensorA[:,[0,1,2]]", sensorA[:,[0,1,2]])
	# print("box[0,2,4]", [box[0,2,4]])	
	# sp2.setData(pos=np.array([0,0,0]),color=[255,140,0,255]) # 成功畫點

	state_string = ["坐", "站", "臥", "跌"]
	try:
		people_state.append(box1[24])
		if len(people_state) == 5: 
			maxlabel = max(people_state, key=people_state.count)
			print("Vote: ", state_string[int(maxlabel)])

			""" Recoding data """
			# Calculate the coordinates (x, y, z)
			x_local = box1[6] + (box1[0] - box1[6]) / 2
			y_local = box1[4] + (box1[1] - box1[4]) / 2
			# z_local = box1[2] + (box1[14] - box1[2]) / 2
			z_local = box1[25]
			# l1_xy = np.sqrt(np.square(x_local) + np.square(y_local))

			# Recoding data (time, x, y, z, state)
			ct3 = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
			with open(path_data, "a",newline="") as csvFile:
				writer = csv.writer(csvFile, dialect = "excel")
				writer.writerow([ct3[11:19], x_local, y_local, z_local, int(maxlabel)])
				# writer.writerow([l1_xy, x_local, y_local, z_local, int(maxlabel), ratio])
			people_state = []

		if maxlabel == 1: #站
			line_color =[89,253,100, 255] 
			width = 6
		elif maxlabel == 0: #坐
			line_color = [0,255,0,255]
			width = 6
		elif maxlabel == 2: #臥
			line_color = [255,255,0,255] 
			width = 6
		elif maxlabel == 3: #fall
			line_color = [255,0,0,255] 
			width = 6

		'''
		[x_max, y_max, z_min, x_max, y_min,z_min, x_min, y_min, z_min,x_min, y_max, z_min,
		x_max, y_max, z_max, x_max, y_min,z_max, x_min, y_min, z_max,x_min, y_max, z_max, state]
		'''
		line1.setData(pos=np.array([box1[0:3], box1[3:6]]),color=line_color, width = width)
		line2.setData(pos=np.array([box1[3:6], box1[6:9]]),color=line_color, width = width)
		line3.setData(pos=np.array([box1[6:9], box1[9:12]]),color=line_color, width = width)
		line4.setData(pos=np.array([box1[9:12], box1[0:3]]),color=line_color, width = width)

		line5.setData(pos=np.array([box1[12:15], box1[15:18]]),color=line_color, width = width)
		line6.setData(pos=np.array([box1[15:18], box1[18:21]]),color=line_color, width = width)
		line7.setData(pos=np.array([box1[18:21], box1[21:24]]),color=line_color, width = width)
		line8.setData(pos=np.array([box1[21:24], box1[12:15]]),color=line_color, width = width)

		line9.setData(pos=np.array([box1[0:3], box1[12:15]]),color=line_color, width = width)
		line10.setData(pos=np.array([box1[3:6], box1[15:18]]),color=line_color, width = width)
		line11.setData(pos=np.array([box1[6:9], box1[18:21]]),color=line_color, width = width)
		line12.setData(pos=np.array([box1[9:12], box1[21:24]]),color=line_color, width = width)

		#line.setData(pos=np.array([box1[0:2], box1[1:3]]),color=[89,253,100, 255])
		"""
		img3.setImage(mapSum)
		"""
	except NameError:
		pass


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(100)

colors = [[255,0,0,255], [0, 255, 0, 255],[248, 89, 253, 255], [89, 253,242, 255],[89, 253,253, 255],
			[253, 89,226, 255],[253, 229,204, 255],[51,255,255, 255],[229,204,255,255], [89,253,100, 255], 
			 [127,255,212, 255], [253,165,89, 255],[255,140,0,255],[255,215,0,255],[0, 0, 255, 255]]

########################################################################
#
# [cloudPoint] -> DBSCAN -> [cluster] -> dispatch Cluster points
#											to Map Array
#	-> [Show Sum of map Array]
#  
########################################################################
mapSizeX = 10
mapSizeY = 10
offSetX = 5.0
state = 0
sensorA = np.empty((100,6))
mapSum = np.zeros((mapSizeX,mapSizeY))

#serialData: ([[x,y,z,range,Doppler,noise,labels]....])
def sensorA2Map(serialData):
	map_10x10 = np.zeros((mapSizeX,mapSizeY))
	for item in serialData:
		#print( "x:{:} y:{:} z:{:}".format(item[0],item[1],item[2]))
		if item[0] < 10 and item[1] < 10: 
			map_10x10[int(item[0] + offSetX),int(item[1])] += 1
	return map_10x10

def single_np(arr, target):
	arr = np.array(arr)
	mask = (arr == target)
	arr_new = arr[mask]
	return arr_new.size


mapA = np.zeros((3,mapSizeX,mapSizeY))
changeable_thread = []
stardand = 1
coco = False

def radarExec():
	global v6len,v7len,v8len,pos1,gcolorA,zOffSet,sensorA,mapA,mapSum,box1, state, before_frame_xy, before_frame_yz, before_frame_zx, queue, coco
	global stardand
	flag = True
	(dck,v6,v7,v8)  = radar.tlvRead(False)

	if dck:
		v8len = len(v8)
		v6len = len(v6)
		v7len = len(v7)

		if v6len != 0 and flag == True:
			queue.append(v6)
			if len(queue) == 4: #這裡數字 = number of frame + 1
				queue.popleft()  

			queue_list = [item for sublist in queue for item in sublist]
			"""
			上面那行的意思
			flat_list = []
			for sublist in t:
				for item in sublist:
					flat_list.append(item)
			"""
			#print("queue", queue_list)
			#print("_____________________________________________________")
			#print("v6", v6)
			flag = False
			pct = queue_list #所有點雲的矩陣
			
			# v6 struct = [(e,a,d,r,sn),(e,a,d,r,sn),(e,a,d,r,sn)..]
	
			# for index in range(len(pct)-1):
			# 	if pct[index][4] < 5:
			# 		pct.pop(index)

			pos1X = np.empty((len(pct),6)) 
			gcolorA = np.empty((len(pct),4), dtype=np.float32)
			
			#print("總共幾個點雲:", len(pct))
			#(1.1) Extract x,y,z,doppler,noise from V6
			
			for i in range(len(pct)):
				zt = pct[i][3] * np.sin(pct[i][0]) + zOffSet
				xt = pct[i][3] * np.cos(pct[i][0]) * np.sin(pct[i][1])
				yt = pct[i][3] * np.cos(pct[i][0]) * np.cos(pct[i][1])
				pos1X[i] = (xt,yt,zt,pct[i][3],pct[i][2],pct[i][4]) # [x,y,z,range,Doppler,noise]

			# SNR 判斷
			""" Pandas
			df = pd.DataFrame(pos1X)
			df2 = df.drop(df[df[:][5] < 8].index)
			if len(df2) == 0:
				df = df.drop(df[df[:][5] < 5].index)
			else:
				df = df2
			pos1X = df.to_numpy()
			"""

			""" numpy """
			df2 = pos1X[pos1X[:, 5] >= 2]
			if len(df2) == 0:
				pos1X = pos1X
			else:
				pos1X = df2

			#(1.2)DBSCAN 預設0.8
			db = DBSCAN(eps=0.6, min_samples=5).fit(pos1X[:,[0,1,2]])
			labels = db.labels_

			#(1.3)insert labels to sensor temp Array(stA) stA = [pos1[X],labels]
			stA = np.insert(pos1X,6,values=labels,axis= 1) #[x,y,z,range,Doppler,noise,labels]

			#(1.4)remove non-cluster point
			mask = (labels == -1)
			sensorA = stA[~mask]
			lbs = labels[~mask]

			"""
			# 找三個方向的點雲相較於上一偵的差異
			if len(sensorA)>0:
				x_y = []
				for i in range(3):
					for xy in sensorA:
						if i == 0:
							x_y.append([round((xy[0]+10)*10), round((xy[1]+10)*10)])
						elif i == 1:
							x_y.append([round((xy[1]+10)*10), round((xy[2]+10)*10)])
						elif i == 2:
							x_y.append([round((xy[2]+10)*10), round((xy[0]+10)*10)])
						#print(x_y)
						d_image = np.zeros((200, 200), dtype = "uint8")
						for pixel in x_y:
							d_image[pixel[1]][pixel[0]] = 255
						
						d_image = cv2.flip(d_image, 1)
						kernel = np.ones((5,5), np.uint8)
						heatmapshow = cv2.dilate(d_image, kernel, iterations = 1)
						heatmapshow = cv2.resize(heatmapshow, (200, 200), interpolation=cv2.INTER_AREA)
					if i == 0:
						before_frame1_xy = heatmapshow.copy()
						before_frame2_xy = cv2.absdiff(before_frame1_xy, before_frame_xy)
						diff_pixel_number_xy = single_np(before_frame2_xy.ravel(), 255)
					elif i == 1:
						before_frame1_yz = heatmapshow.copy()
						before_frame2_yz = cv2.absdiff(before_frame1_yz, before_frame_yz)
						diff_pixel_number_yz = single_np(before_frame2_yz.ravel(), 255)
					elif i == 2:
						before_frame1_zx = heatmapshow.copy()
						before_frame2_zx = cv2.absdiff(before_frame1_zx, before_frame_zx)
						diff_pixel_number_zx = single_np(before_frame2_zx.ravel(), 255)
				before_frame_xy = before_frame1_xy
				before_frame_yz = before_frame1_yz
				before_frame_zx = before_frame1_zx
				'''
				im = np.zeros((200, 1000, 3), dtype = "uint8")
				cv2.rectangle(im, (0, 60), (diff_pixel_number_xy, 80), (0,0,255), -1, cv2.LINE_AA)
				cv2.putText(im, str(diff_pixel_number_xy), (diff_pixel_number_xy+20, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.rectangle(im, (0, 120), (diff_pixel_number_yz, 140), (0,255,0), -1, cv2.LINE_AA)
				cv2.putText(im, str(diff_pixel_number_yz), (diff_pixel_number_yz+20, 120), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.rectangle(im, (0, 180), (diff_pixel_number_zx, 200), (255,0,0), -1, cv2.LINE_AA)
				cv2.putText(im, str(diff_pixel_number_zx), (diff_pixel_number_zx+20, 180), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.imshow("im", im)
				'''
				#cv2.waitKey(0)
			"""

			# 過濾最大集合外的點雲
			if len(sensorA) > 0:
				allpointarray = []
				for each_point in sensorA:
					allpointarray.append(int(each_point[6]))
				mostlabel = np.argmax(np.bincount(np.array(allpointarray)))
				mask1 = (lbs != mostlabel)
				sensorA = sensorA[~mask1]
				lbs = lbs[~mask1]

			# (1.5)assign color to cluster 
			gcolorA = np.empty((len(sensorA),4), dtype=np.float32)
			for i in range(len(lbs)):
				gcolorA = colors[lbs[i]%15]
			
			# (2)get Target Box:
			# get same label id
			# all_point = []
			for k in set(lbs):
				gpMask = (lbs == k)
				#all_point.append(sensorA[gpMask])
				#box0, box1 = get3dBox(all_point)
				if len(sensorA) > 0:
					'''
					box0, box1 = get3dBox(sensorA[gpMask], diff_pixel_number_xy, diff_pixel_number_yz, diff_pixel_number_zx)
					'''
					# box1 = get3dBox(sensorA[gpMask], diff_pixel_number_xy, diff_pixel_number_yz, diff_pixel_number_zx)
					box1 = get3dBox(sensorA[gpMask])
					coco = True
				#print("Get 3D Box: k:{:} box={:}".format(k,box0))
			''' changeable thread
			if coco == True:
				if len(changeable_thread) < 200:
					changeable_thread.append(box1[25])
				else:
					stardand = np.mean(changeable_thread)
			'''
				# stardand = pd.Series(changeable_thread).quantile(0.75)
				# print("stardand", stardand)

			#(3.0)sensorA data mapping to 10x10 map and insert to mapA(map Array)
			# mapA : 10x10x6
			mapA[:-1] = mapA[1:]
			mapA[-1] = sensorA2Map(sensorA)
			
			#(3.1) Sum map array
			# mapsum is data for Plot
			mapSum = np.sum(mapA,axis=0) 
			#print("------------------------------------")
			#print(mapSum.transpose())
			
			#print("labels.count= {:} pos1X= {:} len={:}".format(len(labels),len(pos1X),len(gcolor)))
			#pos1 = sensorA[:,[0,1,2]]
			flag = True

	port.flushInput()
	
def find_2d_boundbox(x):  # x, diff_pixel_number_xy, diff_pixel_number_yz, diff_pixel_number_zx
	global state_fall_lying
	state = 1  # default
	z_mean = np.mean(x[:,2])
	x_Q3 = pd.Series(x[:,0]).quantile(0.75)
	x_Q1 = pd.Series(x[:,0]).quantile(0.25)
	x_IRQ = x_Q3 - x_Q1
	x_max = x_Q3 + 0.5*x_IRQ
	x_min = x_Q1 - 0.5*x_IRQ
	y_Q3 = pd.Series(x[:,1]).quantile(0.75)
	y_Q1 = pd.Series(x[:,1]).quantile(0.25)
	y_IRQ = y_Q3 - y_Q1
	y_max = y_Q3 + 0.5*y_IRQ
	y_min = y_Q1 - 0.5*y_IRQ
	z_Q3 = pd.Series(x[:,2]).quantile(0.75)
	z_Q1 = pd.Series(x[:,2]).quantile(0.25)
	z_IRQ = z_Q3 - z_Q1
	z_max = z_Q3 + 1.5*z_IRQ
	z_min = z_Q1 - 1.5*z_IRQ

	# Smooth
	smooth_xr.append(x_max)
	smooth_xl.append(x_min)
	smooth_yr.append(y_max)
	smooth_yl.append(y_min)
	smooth_zr.append(z_max)
	smooth_zl.append(z_min)
	smooth_zmean.append(z_mean)
	if len(smooth_xr) > 5:
		x_max = np.mean(np.array(smooth_xr.pop(0)))
		x_min = np.mean(np.array(smooth_xl.pop(0)))
		y_max = np.mean(np.array(smooth_yr.pop(0)))
		y_min = np.mean(np.array(smooth_yl.pop(0)))
		z_max = np.mean(np.array(smooth_zr.pop(0)))
		z_min = np.mean(np.array(smooth_zl.pop(0)))
		z_mean = np.mean(np.array(smooth_zmean.pop(0)))

	l1_dis = np.sqrt(np.square(x_min + (x_max - x_min)/2) + np.square(y_min + (y_max - y_min)/2))
	error_from_radar = randomForestModel.predict(np.array(l1_dis).reshape(-1, 1))
	z_mean = z_mean - error_from_radar[0]
	z_fall_lying.append(z_mean)
	if len(z_fall_lying) > 10:
		z_fall_lying.pop(0)
	if z_mean > 0.8:  # z_men > 0.8 (reset to 0)
		state_fall_lying = 0
	else:
		if len(z_fall_lying) > 9:
			tmp_threshold = np.mean(z_fall_lying[:5])
			dif_z_thr = np.abs(tmp_threshold - z_mean)
			if dif_z_thr > 0.2:  # add: 放寬閾值
				state_fall_lying = 1

	lenofz = z_max - z_min
	lenofx = x_max - x_min
	lenofy = y_max - y_min

	if lenofz/lenofx >= 1 or lenofz/lenofy >= 1 or lenofx == 0 or lenofy == 0:  # add: lenofx == 0 or lenofy == 0 (除 0 error)
		if z_mean > (stardand / 10) * 8:
			state = 1 # 站
		elif lenofz/lenofx < 0.85 or lenofz/lenofy < 0.85:  # defalt = 0.6
			if state_fall_lying == 0:
				state = 2 # 臥
			else:
				state = 3 # 跌
		else:
			state = 0
	else:
		if state_fall_lying == 0:
			state = 2 # 臥
		else:
			state = 3 # 跌
		
	return np.array([x_max, y_max, z_min, x_max, y_min,z_min, x_min, y_min, z_min,x_min, y_max, z_min,
					 x_max, y_max, z_max, x_max, y_min,z_max, x_min, y_min, z_max,x_min, y_max, z_max, state, z_mean])

def get3dBox(targetCloud):  # targetCloud, diff_pixel_number_xy, diff_pixel_number_yz, diff_pixel_number_zx
	box = find_2d_boundbox(targetCloud[:,0:3])  # targetCloud[:,0:3], diff_pixel_number_xy, diff_pixel_number_yz, diff_pixel_number_zx
	'''
	# x axis
	xMax = np.max(targetCloud[:,0])
	xr   = np.min(targetCloud[:,0])
	xl = np.abs(xMax-xr)

	# y axis
	yMax = np.max(targetCloud[:,1])
	yr = np.min(targetCloud[:,1])
	yl = np.abs(yMax-yr)

	# z axis
	zMax = np.max(targetCloud[:,2])
	zr = np.min(targetCloud[:,2])
	zl = np.abs(zMax-zr)
	return np.array([xr,xl,yr,yl,zr,zl]), box
	'''
	return box
	
def uartThread(name):
	port.flushInput()
	while True:
		radarExec()
					
thread1 = Thread(target = uartThread, args =("UART",))
thread1.setDaemon(True)
thread1.start()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    before_frame_xy = np.zeros((200, 200), np.uint8)
    before_frame_yz = np.zeros((200, 200), np.uint8)
    before_frame_zx = np.zeros((200, 200), np.uint8)
    queue = deque([])
    if (sys.flags.interactive != 1) or not hasattr(QtCore,'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
