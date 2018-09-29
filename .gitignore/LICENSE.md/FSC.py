import cv2
import numpy as np
import random
from matplotlib import pyplot as plot

img1=cv2.imread("D:\img\one.png",0)
img2=cv2.imread("D:\img\on.png",0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
#print(len(kp1))
#print(len(kp2))
print(np.array(kp1[3].pt))

bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
matches=bf.match(des1,des2)
#matches=sorted(matches,key=lambda x:x.distance)
print(len(matches))
img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
plot.imshow(img3),plot.show()

bf1 = cv2.BFMatcher()
matcher = bf1.knnMatch(des1,des2, k=2)
good_matcher=[]
for m,n in matcher:
    if m.distance < 0.75*n.distance:
        good_matcher.append(m)
        
img4=cv2.drawMatches(img1,kp1,img2,kp2,good_matcher[:10],None,flags=2)
plot.imshow(img4),plot.show()
print(len(good_matcher))

def rand_index():  
    index=[]
    while(len(index)<3):
        x=random.randint(0,len(good_matcher))
        if x not in index:
            index.append(x)
    return index
index=rand_index()
for i in index:
    a=kp1[good_matcher[i].queryIdx].pt
    b=kp2[good_matcher[i].trainIdx].pt
    print(np.array(a))
#print(good_matcher[3].queryIdx)
#print(good_matcher[3].trainIdx)
def parameter(sets):
    query=[]
    train=[]
    index=rand_index()
#    print(index)
    for i in index:
        a=kp1[sets[i].queryIdx].pt
        b=kp2[sets[i].trainIdx].pt
        query.append(np.array(a))
        train.append(np.array(b))
    query=np.array(query)
    train=np.array(train)
#print(query)
#print(train)
    pts1 = np.float32(query)
    pts2 = np.float32(train)
    M = cv2.getAffineTransform(pts1,pts2)
    return M
#print(M)
N=0
for i in range(300):
    par=parameter(good_matcher)
    Ci=[]
    for j in range(len(matches)):
        c=np.array(kp1[matches[j].queryIdx].pt)
        d=np.array(kp2[matches[j].trainIdx].pt)
        g=np.append(c,1)
        f=np.dot(par,g.T)
        dist=np.linalg.norm(d-f)
        if dist<1:
            Ci.append(matches[j])
    if len(Ci)>N:
        N=len(Ci)
print(len(Ci))
par2=parameter(Ci)  
print(par2)
Smod=[]
angle=[]
for i in range(len(Ci)):
    a=kp1[Ci[i].queryIdx].octave
    b=kp2[Ci[i].trainIdx].octave
    d=kp1[Ci[i].queryIdx].angle
    e=kp2[Ci[i].trainIdx].angle
    c=a/b
    f=d-e
    Smod.append(c)
    angle.append(f)
#print(Smod)
#print(angle)
plot.hist(angle,density=True, facecolor='g')
plot.show()
