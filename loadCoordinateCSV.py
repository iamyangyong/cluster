import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib import image
import cv2
import math as ma
import random

# get cluster in every frame
# pixelSize = pysical size/magnification factor
pixelSize = 0.11
# ramdon drift distance
# drift = 0.1
#  minimun number of points in a cluster
minpts = 5
# distance threshold
# epsilon = (drift/pixelSize)*sqrt(minpts);
epsilon = 35

def loadCoordinateFromCSV(filepath,filename):
    # 原始数据记录
    originTracks = []
    # 按照trackId分组并按照frame升序后(t,x,y)
    validTracks = []
    # 有效索引
    validIndex = []

    pathlist = glob.glob(filepath + filename)
    # 文件路径+文件名
    for csvfilepath in pathlist:
        # 文件名
        csvfilename= os.path.basename(csvfilepath)
        fio_raw=pd.read_csv(csvfilepath)
        # 按照FRAME列排序
        fio_raw.sort_values(by='FRAME',inplace=True,ascending=True)
        # 取出所有TRACK_ID并去重然后升序
        track_ids=sorted(list(set(fio_raw['TRACK_ID']))) #TRACK_ID
        # 按照TRACK_ID分组
        for trackid in track_ids:
                trackItems=fio_raw[fio_raw['TRACK_ID']==trackid]#TRACK_ID
                if len(trackItems)>0:
                    trackItem=trackItems.values
                    originList=[]
                    for item in trackItem:
                        #  让帧从1开始计数
                        t = int(item[8])+1
                        x = float(item[4])*9.0909
                        y = float(item[5])*9.0909
                        originList.append((t,x,y))
                originTracks.append(originList) #原始数据

        for tracks in originTracks:
                if len(tracks)>=15:
                    validIndex.append(originTracks.index(tracks))
                    validTracks.append(tracks) #有连续15帧及以上的数据
    return originTracks,validTracks,validIndex

# 检查粒子是否在当前帧中
def getCoordinateByFrame(validTracks):
    everyParticleFrameCount = []
    dataByFrame = []
    particleList=[]
    len_validTracks=0
    for particle in validTracks:
        everyParticleFrameCount.append(len(particle))
        len_validTracks+=1
    maxFrameNum = max(everyParticleFrameCount)
    #  遍历所有帧
    k=0
    for i in range(0,maxFrameNum):
        count=0
        k+=1
        particleList=[]
        for particles in validTracks:
            for particle in particles:
                if k == particle[0]:
                    count+=1
                    particleList.append((count,particle[1],particle[2]))
        dataByFrame.append(particleList)
    return dataByFrame,maxFrameNum

# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = ma.sqrt((np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2)))
    # print("两点之间的距离为："+str(dis))
    return dis

def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):# and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k+1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi])<=Eps): #and (j!=pi):
                            M.append(j)
                    if len(M)>=MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
    return C

# 聚类算法
def algorithmsFun(x,epsilon,minpts,i):
    y_pred = DBSCAN(eps = 10, min_samples = 5).fit_predict(x)
    # plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    return y_pred

#聚类操作
def cluster(dataByFrame,maxFrameNum):
    clusterdata = np.empty((len(dataByFrame),1),dtype=object)
    c=0
    count=0
    for dataFrame in dataByFrame:
        pointData=np.empty((len(dataFrame),2),dtype=object)
        k=0
        for data in dataFrame:
            pointData[k,0]=data[1]
            pointData[k,1]=data[2]
            k+=1
        # idx=algorithmsFun(pointData,epsilon,minpts,count)
        idx = dbscan(pointData,epsilon,minpts)
        count+=1
        clusterdata[c,0]=idx
        c+=1
    return clusterdata

# 计算点云外轮廓围城的面积
def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i+1)%n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area,n

#绘制每一帧粒子的点和这些点围城的边框
def plotParticleEveryFrameInCluster(maxFrameNum,dataByFrame,clusterData):
    # 获取所有图像的路径
    imageSeqName=[]
    for root, dirs, files in os.walk(filepath + r'/imageSeq/'):
        for name in files:
            imageSeqName.append(os.path.join(root, name))
    imageSeqName.sort()
    
    particlePathName = os.path.join(filepath,"particleResultInCluster/")
    if not os.path.isdir(particlePathName):
        os.mkdir(particlePathName)
    
    # 每一帧聚类结果
    for i in range(0,maxFrameNum):
        print("当前第%d帧，总帧数：%d"%(i,maxFrameNum))
        imgName = imageSeqName[i]
        img = cv2.imread(imgName)
        # imgs = image.imread(imgName)
        pointList=[]
        particleIDList=[]
        for data in dataByFrame[i]:
            particleID = data[0]
            x = int(float(data[1]))
            y = int(float(data[2]))
            particleIDList.append(particleID)
            pointList.append((x,y))
        print("第%d帧粒子数：%d"%(i,len(pointList)))

        # 在原图上描点
        for point in pointList:
            cv2.circle(img,point,1,(0,0,255),thickness=3)
        imagefilename = filepath + "clusterImage/{}.jpg".format(str(i))
        cv2.imwrite(imagefilename,img)

        # 寻找聚集点
        clusterNum = max(clusterData[i][0])
        if clusterNum >= 0:
            for k in range(0,clusterNum+1):
                clusterbyframe = np.array(clusterData[i][0],dtype=np.int32)
                clusterIdx = np.where(clusterbyframe==k)
                dataCusterPoint1=[]
                for index  in clusterIdx[0]:
                    dataCusterPoint1.append((pointList[index][0],pointList[index][1]))
                dataCusterPoint2 = np.array(dataCusterPoint1,dtype = np.int32)
                hull = cv2.convexHull(dataCusterPoint2)
                cv2.polylines(img,[hull],True,(0,0,255),2)
                po = []
                for mk in range(len(hull)):
                    po.append((hull[mk][0][0],hull[mk][0][1]))
                area,n = polygon_area(po)
                print("第%d帧的第%d组粒子围成的面积是：%d,边界上的粒子数是：%d,边界内的粒子数是：%d"%(i,k,area,n,len(dataCusterPoint2)-n))
                dataCusterPoint2 = None
            imagefilename = filepath + "particleResultInCluster/{}.jpg".format(str(i))
            cv2.imwrite(imagefilename,img)
        break

if __name__ == "__main__":
    originTracks,validTracks,validIndex = loadCoordinateFromCSV(filepath='/ldap_shared/home/v_yy/Project/MoTT/cluster/test4/',filename='test.csv')
    dataByFrame,maxFrameNum = getCoordinateByFrame(validTracks)
    clusterData = cluster(dataByFrame,maxFrameNum)
    plotParticleEveryFrameInCluster(maxFrameNum,dataByFrame,clusterData,filepath='/ldap_shared/home/v_yy/Project/MoTT/cluster/test4/')
