import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
import matplotlib.patches as patches
import skimage.io as io
import time
import os
data_dir = './data'
train_list = []
val_list = []
name_list = []
bounding_box = []

dataset = 'cub_200'

for line in open(os.path.join(data_dir, dataset, 'train.txt'), 'r'):
    train_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
for line in open(os.path.join(data_dir, dataset, 'val.txt'), 'r'):
    val_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))

for line in open(os.path.join(data_dir, dataset, 'classes.txt'), 'r'):
    name_list.append(line.strip().split(' ')[1])

for line in open(os.path.join(data_dir, dataset, 'bounding_boxes.txt'), 'r'):
   bounding_box.append(line.strip())

print("tets "+str(val_list[500][1]))
box = bounding_box[val_list[500][1]+1].split(" ")
print(box[0]+" "+box[1]+" "+box[2]+" "+box[3]+" "+box[4])


w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 3
rows =3
for i in range(6):
   I = io.imread(val_list[500][0])
   fig.add_subplot(200, 200, i)		
   figs,ax = plt.subplots(1)
   plt.axis('off')
   plt.imshow(I)
   plt.suptitle(os.path.basename(val_list[500][0]), fontsize=10)
   rect = patches.Rectangle((float(box[1]),float(box[2])),float(box[3]),float(box[4]),linewidth=1,edgecolor='r',facecolor='none')
   ax.add_patch(rect)

plt.show()