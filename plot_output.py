import matplotlib.pyplot as plt;
import pandas as pd;
import distinctipy as ds;
import sys

FILE=pd.read_csv("output.csv")
if len(FILE.columns)>4:
	sys.exit("Cannot plot more than 3D data")
x=FILE.iloc[:,0]
y=FILE.iloc[:,1]
if len(FILE.columns)==4:
	z=FILE.iloc[:,2]
	ax = plt.axes(projection ="3d");
labels=FILE.iloc[:,-1:]
noofcolors=int(labels.max())+1
colors=ds.get_colors(noofcolors)
cvec = [colors[int(label[1])] for label in labels.iterrows()]
if len(FILE.columns)==3:
	plt.scatter(x,y,c=cvec,s=8)
if len(FILE.columns)==4:
	ax.scatter3D(x,y,z,c=cvec,s=8)
plt.savefig("data.png")
plt.show()
