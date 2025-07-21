
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import datetime

import numpy as np

from tkinter.filedialog import askopenfilename

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
import skimage.io as io
import time
import os
import matplotlib.patches as patches



main = tkinter.Tk()
main.title("Fine-Grained Image Classification Using Modified DCNNs Trained by Cascaded Softmax and Generalized Large-Margin Losses")
main.geometry("1200x1200")

global filename

LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)

pretrain_model = 'inception_v3_iNat_299'
dataset = 'cub_200'

load_dir = os.path.join('./feature', pretrain_model)
features_train = np.load(os.path.join(load_dir, dataset + '_feature_train.npy'))
labels_train = np.load(os.path.join(load_dir, dataset + '_label_train.npy'))
features_val = np.load(os.path.join(load_dir, dataset + '_feature_val.npy'))
labels_val = np.load(os.path.join(load_dir, dataset + '_label_val.npy'))
print(features_train.shape)
print(labels_train.shape)
print(features_val.shape)
print(labels_val.shape)

tic = time.time()
LR.fit(features_train, labels_train)
labels_pred = LR.predict(features_val)

num_class = len(np.unique(labels_train))
acc = np.zeros((num_class, num_class), dtype=np.float32)
for i in range(len(labels_val)):
    acc[int(labels_val[i]), int(labels_pred[i])] += 1.0

fig, ax = plt.subplots(figsize=(6,6))
plt.imshow(acc)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)

print('Accuracy: %f' % (sum([acc[i,i] for i in range(num_class)]) / len(labels_val)))
print('Elapsed Time: %f s' % (time.time() - tic))

data_dir = './data'
train_list = []
val_list = []
bounding_box = []
for line in open(os.path.join(data_dir, dataset, 'train.txt'), 'r'):
    train_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
for line in open(os.path.join(data_dir, dataset, 'val.txt'), 'r'):
    val_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
for line in open(os.path.join(data_dir, dataset, 'bounding_boxes.txt'), 'r'):
   bounding_box.append(line.strip())



def upload():
   
    global filename
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text=filename)

def DCNN():
    name = os.path.basename(filename)
    arr = name.split(".")
    kdt = KDTree(features_train, leaf_size=30, metric='euclidean')
    K = 5
    q_ind = int(arr[0])
    box = bounding_box[val_list[q_ind][1]+1].split(" ")
    print(features_val[q_ind:q_ind+1])
    dist, ind = kdt.query(features_val[q_ind:q_ind+1], k=K)
    print('Query image from validation set:')
    I = io.imread(filename)
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(I)
    plt.suptitle("query image : "+os.path.basename(filename), fontsize=10)
    #rect = patches.Rectangle((float(box[1]),float(box[2])),float(box[3]),float(box[4]),linewidth=1,edgecolor='r',facecolor='none')
    #ax.add_patch(rect)
    plt.show()

    
    for i in range(K):
       plt.figure(figsize=(30,30))
       plt.subplot(1, K, i+1)
       I = io.imread(train_list[ind[0,i]][0])
       box = bounding_box[train_list[ind[0,i]][1]+1].split(" ")
       fig,ax = plt.subplots(1)
       plt.axis('off')
       plt.imshow(I)
       plt.suptitle(os.path.basename(train_list[ind[0,i]][0]), fontsize=10)
       rect = patches.Rectangle((float(box[1]),float(box[2])),float(box[3]),float(box[4]),linewidth=1,edgecolor='r',facecolor='none')
       ax.add_patch(rect)
    plt.show()

    

font = ('times', 20, 'bold')
title = Label(main, text='Bird species classification')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

depthbutton = Button(main, text="Run DCNN Algorithm", command=DCNN)
depthbutton.place(x=50,y=150)
depthbutton.config(font=font1) 




main.config(bg='brown')
main.mainloop()