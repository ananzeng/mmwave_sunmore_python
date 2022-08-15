import os
from re import T
from symtable import Symbol
import sys
from typing import no_type_check
import serial
#import hdbscan
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from mmWave import pc3
from threading import Thread
from collections import deque
from sklearn.cluster import DBSCAN
from pyqtgraph.Qt import QtCore, QtGui

import warnings
warnings.filterwarnings("ignore")

coco = True
coco_show = False
normal_flag = False
"""
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

app = QtGui.QApplication([])
w = gl.GLViewWidget()
# w = pg.PlotWidget()
w.show()

####### camera position #######
w.setCameraPosition(distance=7, elevation=50, azimuth=90)

#size=50:50:50
g = gl.GLGridItem()
g.setSize(x=50,y=50,z=50)
#g.setSpacing(x=1, y=1, z=1, spacing=None)
w.addItem(g)

####### draw axis ######
axis = Custom3DAxis(w, color=(0.2,0.2,0.2,1.0))
axis.setSize(x=25, y=25, z=25)
xt = [0,5,10,15,20,25]  
axis.add_tick_values(xticks=xt, yticks=xt, zticks=xt)
w.addItem(axis)
w.setWindowTitle('Position Occupancy(Cluster)')

port = serial.Serial("COM3",baudrate = 921600 , timeout = 0.5)
radar = pc3.Pc3(port)

v6len = 0
v7len = 0
v8len = 0

pos = np.zeros((100,3))
color = [1.0, 0.0, 0.0, 1.0]
sp1 = gl.GLScatterPlotItem(pos = pos, color = color, size = 15.0)
w.addItem(sp1)

gcolorA = np.empty((100,4), dtype = np.float32)

def update():
	global gcolorA, sensorA, coco_show
	#extract Labels
	#print("labels len:{:}".format(sensorA.shape))
	if coco_show:
		# sp1.setData(pos=sensorA[:, [0,1,2]], color = gcolorA)  # 原版
		if len(pre_center) >= 1:
			tmp = np.array(pre_center)
			tmp[:, 2] = 0
			sp1.setData(pos=tmp, color = [255,0,0,255])  # 改版  color = gcolorA
			coco_show = False

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(100)

colors = [[255,0,0,255], [0, 255, 0, 255],[248, 89, 253, 255], [89, 253,242, 255],[89, 253,253, 255],
			[253, 89,226, 255],[253, 229,204, 255],[51,255,255, 255],[229,204,255,255], [89,253,100, 255], 
			 [127,255,212, 255], [253,165,89, 255],[255,140,0,255],[255,215,0,255],[0, 0, 255, 255]]
"""

""" 2D """
#port = serial.Serial("COM6",baudrate = 921600 , timeout = 0.5)
port = serial.Serial("/dev/ttyTHS1", baudrate=921600, timeout=0.5)
radar = pc3.Pc3(port)

colors = [[255,0,0,255], [0, 255, 0, 255],[248, 89, 253, 255], [89, 253,242, 255],[89, 253,253, 255],
			[253, 89,226, 255],[253, 229,204, 255],[51,255,255, 255],[229,204,255,255], [89,253,100, 255], 
			[127,255,212, 255], [253,165,89, 255],[255,140,0,255],[255,215,0,255],[0, 0, 255, 255]]

## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])
w = QtGui.QWidget()
w.resize(1300, 700)

## Create some widgets to be placed inside
listw = QtGui.QListWidget()
listw.setFixedWidth(260)
plot = pg.PlotWidget()
plot.setXRange(-5, 5)
plot.setYRange(0, 10)
## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

# Create the scatter plot and add it to the view
pen = pg.mkPen(width=2, color='r')
scatter = pg.ScatterPlotItem(pen=pen, symbol='o', size=12)
plot.addItem(scatter)

# ID
text2 = {i: pg.TextItem("", anchor=(0.5, -1.0)) for i in range(1000)}
for i in range(1000):
	text2[i].setParentItem(scatter)

## Add widgets to the layout in their proper positions
layout.addWidget(listw, 1, 0)  # list widget goes in bottom-left
layout.addWidget(plot, 1, 1, 1, 1)  # plot goes on right side, spanning 3 rows

## Display the widget as a new window
w.setWindowTitle('Position Occupancy(Cluster)')
w.show()

def update():
	global gcolorA, sensorA, coco_show, text2
	if coco_show:
		# sp1.setData(pos=sensorA[:, [0,1,2]], color = gcolorA)  # 原版
		if len(pre_center) >= 1:
			tmp = np.array(pre_center)
			tmp[:, 0] = -tmp[:, 0]  # [[x, y, id, state], [x, y, id, state], ...]
			
			scatter.setData(pos=tmp, color = [255,0,0,255])  # 改版  color = gcolorA
			
			if coco != True:
				listw.clear()
				for i in range(len(text2)):
					text2[i].setText("")
			if normal_flag == True:  # 當為正常資料才會預測狀態
				listw.addItem(f"數量: {len(pre_center)}")
				
				# text2 = {i: pg.TextItem("", anchor=(0.5, -1.0)) for i in range(len(pre_center))}
				for i in range(len(pre_center)):
					if len(tmp) > 0:
						if tmp[i, 3] == 0:
							state = "停止"
						elif tmp[i, 3] == 1:
							state = "慢移"
						elif tmp[i, 3] == 2:
							state = "快移"
						else:
							state = "慢移"

						listw.addItem("ID:{}  X: {:.3f}  Y: {:.3f}  State: {}".format(int(tmp[i, 2]), tmp[i, 0], tmp[i, 1], state))
						text2[i].setParentItem(scatter)
						text2[i].setPos(tmp[i, 0], tmp[i, 1])
						text2[i].setText(str(int(tmp[i, 2])))
				
				# for i in range(len(pre_center)):
				# text2.hide()
				# plot.removeItem(text2)
			coco_show = False

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(100)

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

sensorA = np.empty((10000,6))
pre_center = []
def radarExec():
	global v6len,v7len,v8len,pos1,gcolorA,sensorA, coco, current_center, tmp_pre_center, coco_show, normal_flag, pre_center
	flag = True
	(dck,v6,v7,v8)  = radar.tlvRead(False)

	if dck:
		v8len = len(v8)
		v6len = len(v6)
		v7len = len(v7)
		
		#print("Sensor Data: [v6,v7,v8]:[{:d},{:d},{:d}]".format(v6len,v7len,v8len))
		if v6len != 0 and flag == True:
			flag = False
			
			""" 累積前幾個 Frame """
			queue.append(v6)
			if len(queue) == 3: #這裡數字 = number of frame + 1
				queue.popleft()  
			queue_list = [item for sublist in queue for item in sublist]
			pct = queue_list

			""" 不累積 """
			# pct = v6  # v6 struct = [(e,a,d,r,sn),(e,a,d,r,sn),(e,a,d,r,sn)..]

			pos1X = np.empty((len(pct),6)) 
			gcolorA = np.empty((len(pct),4), dtype=np.float32)
			
			#(1.1) Extract x,y,z,doppler,noise from V6
			for i in range(len(pct)):
				zt = 0
				xt = pct[i][3] * np.cos(pct[i][0]) * np.sin(pct[i][1])
				yt = pct[i][3] * np.cos(pct[i][0]) * np.cos(pct[i][1])
				pos1X[i] = (xt,yt,zt,pct[i][3],pct[i][2],pct[i][4]) #[x,y,z,range,Doppler,noise]
			
			# sn 濾雜訊
			df = pd.DataFrame(pos1X)
			df_droped = df.drop(df[df[:][5] < 8].index)
			if len(df_droped) == 0:
				df = df.drop(df[df[:][5] < 2].index)
			else:
				df = df_droped
			pos1X = df.to_numpy()

			#(1.2)DBSCAN 
			db = DBSCAN(eps=0.5, min_samples=18).fit(pos1X[:,[0,1,2]])  # dbscan  min_samples=8
            # db = hdbscan.HDBSCAN(min_cluster_size=8).fit(pos1X[:,[0,1]])  # hdbscan
			labels = db.labels_
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			# print(f'Estimated number of clusters: --------------------------------------------- {n_clusters_} --------------------------------------------- ')
			'''
			# Number of clusters in labels, ignoring noise if present.
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			print('Estimated number of clusters: %d' % n_clusters_)
			n_noise_ = list(labels).count(-1)
			print('Estimated number of noise points: %d' % n_noise_)
			labelSet = set(labels)
			print("Label Set:{:}".format(labelSet))
			'''
			
			#(1.3)insert labels to sensor temp Array(stA) stA = [pos1[X],labels]
			stA = np.insert(pos1X,6,values=labels,axis= 1) #[x,y,z,range,Doppler,noise,labels]
			
			#(1.4)remove non-cluster point
			mask = (labels == -1)
			sensorA = stA[~mask]
			lbs = labels[~mask]

			""" 算法 """
			if n_clusters_ != 0:
				# 跑第一次
				if coco == True:
					for i in range(n_clusters_):
						cx, cy = (np.mean(sensorA[sensorA[:,6] == i][:,0]), np.mean(sensorA[sensorA[:,6] == i][:,1]))
						current_center = np.array([cx, cy])
						tmp_pre_center = np.array(pre_center)
						pre_center.append([cx, cy, i, 3])  # 例外
						# print(f"center_x: {cx}, center_y: {cy}, id: {i}")
					coco = False

				# 跑第其他次
				else:
					if n_clusters_ == len(pre_center) and n_clusters_ != 0:  # 正常
						for i in range(n_clusters_):

							""" 離群值濾除 (開始) """
							n = 1.5
							# Sort
							current_x = np.sort(sensorA[sensorA[:,6] == i][:,0])
							current_y = np.sort(sensorA[sensorA[:,6] == i][:,1])

							# IQR = Q3 - Q1
							IQR_x = np.percentile(current_x, 75) - np.percentile(current_x, 25)
							IQR_y = np.percentile(current_y, 75) - np.percentile(current_y, 25)

							# outlier = Q3 + n * IQR 
							transform_data_x = current_x[current_x < np.percentile(current_x, 75) + n * IQR_x]
							transform_data_y = current_y[current_y < np.percentile(current_y, 75) + n * IQR_y]

							# outlier = Q1 - n * IQR 
							transform_data_x = transform_data_x[transform_data_x > np.percentile(transform_data_x, 25) - n * IQR_x]
							transform_data_y = transform_data_y[transform_data_y > np.percentile(transform_data_y, 25) - n * IQR_y]
							""" 離群值濾除 (結束) """

							cx, cy = (np.mean(transform_data_x), np.mean(transform_data_y))  # 平均後找 X 中心點跟 Y 中心點
							# cx, cy = (np.mean(sensorA[sensorA[:,6] == i][:,0]), np.mean(sensorA[sensorA[:,6] == i][:,1]))  # 平均後找 X 中心點跟 Y 中心點
							current_center = np.array([cx, cy])  # 平均後找 X 中心點跟 Y 中心點
							tmp_pre_center = np.array(pre_center)
							dist = np.sqrt(np.sum(np.square(current_center - tmp_pre_center[tmp_pre_center[:,2] == i][0][:2])))
							# print(f"dist: {dist}")
							if dist <= 0.08:
								pre_center.append([cx, cy, i, 0])  # 停止
							elif dist > 0.08 and dist < 0.3:
								pre_center.append([cx, cy, i, 1])  # 慢移
							else:
								pre_center.append([cx, cy, i, 2])  # 快移
							# print(f"center_x: {cx}, center_y: {cy}, id: {i}")
							# pre_center.append([cx, cy, i])
							normal_flag = True

					else:  # 不正常
						for i in range(n_clusters_):
							cx, cy = (np.mean(sensorA[sensorA[:,6] == i][:,0]), np.mean(sensorA[sensorA[:,6] == i][:,1]))
							current_center = np.array([cx, cy])
							tmp_pre_center = np.array(pre_center)
							# print(f"center_x: {cx}, center_y: {cy}, id: {i}")
							pre_center.append([cx, cy, i, 3])  # 例外
			else:
				coco = True
			# 刪除上一個 Frame，維持 pre_center 長度
			while len(pre_center) > n_clusters_:
				pre_center.pop(0)

			""" 算法 """
			coco_show = True  # 當 coco_show = True 時, 代表中心點的累積已經完成可以顯示位置了

			#(1.5)assign color to cluster 
			gcolorA = np.empty((len(sensorA),4), dtype=np.float32)
			for i in range(len(lbs)):
				gcolorA = colors[lbs[i]%15]
			
			#print("labels.count= {:} pos1X= {:} len={:}".format(len(labels),len(pos1X),len(gcolor)))
			#pos1 = sensorA[:,[0,1,2]]
		flag = True
		 
	port.flushInput()
	
def uartThread(name):
	port.flushInput()
	while True:
		radarExec()
					
thread1 = Thread(target = uartThread, args =("UART",))
thread1.setDaemon(True)
thread1.start()

if __name__ == '__main__':
    import sys
    queue = deque([])
    if (sys.flags.interactive != 1) or not hasattr(QtCore,'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
