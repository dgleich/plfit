import sys
from sys import path
path.append("/homes/eikmeier/plfit/src")
import numpy as np
import os
import re
import csv
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt






def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)



def grab_vals(short_name, root):
    a1 = None
    a2 = None
    p1 = None
    p2 = None
    resamplealp = None
    resamplep = None
    eig_file = os.path.join(root, short_name+'.adjacency.plfit')
    deg_file = os.path.join(root,short_name+'.plfit')
    orig_degs = os.path.join(root,short_name+'.degs')
    orig_eigs = os.path.join(root, short_name+'.adjacency.eigs')
    resample_file = os.path.join(root, short_name+'.resample.adjacency.plfit')
    for line in open(eig_file, 'r'):
        if 'alpha' in line:
            a1 = re.sub('alpha = ', '', line)
        elif 'p     = ' in  line:
            p1 = re.sub('p     = ', '', line)
    for line in open(deg_file, 'r'):
        if 'alpha' in line:
            a2 = re.sub('alpha = ', '', line)
        elif 'p     = ' in  line:
            p2 = re.sub('p     = ', '', line)
    with open(orig_degs, 'r') as f:
        first_line = f.readline()
    with open(orig_eigs, 'r') as f:
        eigs = [float(line) for line in f]
        max_eig = max(eigs)
    if os.path.exists(resample_file):
        for line in open(resample_file, 'r'):
            if 'alpha' in line:
                resamplealp = re.sub('alpha = ', '', line)
            elif 'p     = ' in line:
                resamplep= re.sub('p     = ', '', line)
    a1 = 0.0 if a1 is None else a1
    a2 = 0.0 if a2 is None else a2
    p1 = 0.0 if p1 is None else p1
    p2 = 0.0 if p2 is None else p2
    resamplealp = 0.0 if resamplealp is None else resamplealp
    resamplep = 0.0 if resamplep is None else resamplep
    return (float(a1), float(p1), float(a2), float(p2), float(first_line), max_eig, float(resamplealp), float(resamplep))
    


#########################
## PHASE 1 
## GATHER THE INFO FROM THE PLFIT FILES
#########################

my_list = []
model_list = []
oregon_list = []
as_list = []
rpl_list = []
p2p_list = []
names_of_files = []
for root, dirs, files in os.walk("/homes/eikmeier/plfit/data"):
    for file in files:
        if file.endswith(".adjacency.plfit") and not file.endswith(".resample.adjacency.plfit"):
            short_name = re.sub('.adjacency.plfit$', '', file)
            deg_file = os.path.join(root,short_name+'.plfit')

            if any(c in file for c in ('copy', 'ff', 'pa')):
                if os.path.exists(deg_file):
                    model_list.append(grab_vals(short_name, root))

            elif any(c in file for c in ('as19', 'as20', 'as-caida')):
                if os.path.exists(deg_file):
                    as_list.append(grab_vals(short_name, root))

            elif 'oregon' in file:
                if os.path.exists(deg_file):
                    oregon_list.append(grab_vals(short_name, root))

            elif 'rpl' in file:
                if os.path.exists(deg_file):
                    rpl_list.append(grab_vals(short_name, root))

            elif 'p2p' in file:
                if os.path.exists(deg_file):
                    p2p_list.append(grab_vals(short_name, root))

            elif os.path.exists(deg_file):
                my_list.append(grab_vals(short_name, root))
                names_of_files.append(short_name)


#Take averages of oregon, rpl, and p2p to include in "my_list"
oregon_array = np.transpose(np.asarray(oregon_list))
p2p_array = np.transpose(np.asarray(p2p_list))
as_array = np.transpose(np.asarray(as_list))
rpl_array = np.transpose(np.asarray(rpl_list))
model_array = np.transpose(np.asarray(model_list))

oregon_avg = [mean(oregon_array[0]), mean(oregon_array[1]), mean(oregon_array[2]), mean(oregon_array[3]), mean(oregon_array[4]), mean(oregon_array[5]), mean(oregon_array[6]), mean(oregon_array[7])]
p2p_avg = [mean(p2p_array[0]), mean(p2p_array[1]), mean(p2p_array[2]), mean(p2p_array[3]), mean(p2p_array[4]), mean(p2p_array[5]), mean(p2p_array[6]), mean(p2p_array[7])]
as_avg = [mean(as_array[0]), mean(as_array[1]), mean(as_array[2]), mean(as_array[3]), mean(as_array[4]), mean(as_array[5]), mean(as_array[6]), mean(as_array[7])]
rpl_avg = [mean(rpl_array[0]), mean(rpl_array[1]), mean(rpl_array[2]), mean(rpl_array[3]), mean(rpl_array[4]), mean(rpl_array[5]), mean(rpl_array[6]), mean(rpl_array[7])]
my_list.append(oregon_avg)
my_list.append(p2p_avg)
my_list.append(as_avg)
my_list.append(rpl_avg)
names_of_files.append('oregon_avg')
names_of_files.append('p2p_avg')
names_of_files.append('as_avg')
names_of_files.append('rpl_avg')

# "my_array" includes every graph and the averages of the three classes listed above
my_array = np.transpose(np.asarray(my_list))


print(oregon_avg)
print(p2p_avg)
print(as_avg)
print(rpl_avg)

#########################
## PHASE 2
## ANALYZE THE DATA
#########################  


####EVERYTHING
eig_idx = np.where(my_array[1] >= 0.1)[0]
deg_idx = np.where(my_array[3] >= 0.1)[0]
idx = [val for val in eig_idx if val in deg_idx]
print('Total number of graphs analyzing: ', len(my_array[1]))
print('Number of power-laws in eigs: ', len(eig_idx))
print('Number of power-laws in degs: ', len(deg_idx))
print('Number of power-laws in both: ', len(idx))


#Make a plot of the alpha-eig vs alpha-deg 
x = (my_array[2])[idx] #degs
y = (my_array[0])[idx] #eigs

#Find a Least-Squares fit to the data
A = np.vstack([x,np.ones(len(x))]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y,label = 'Original Data')
plt.plot(x,m*x+c, 'r', label = 'Fitted Line')
print('m = ',m)
print('c = ',c)
p = np.vstack([m,c])
r = y - A.dot(p)
R2 = 1.0 - np.square( (np.linalg.norm(r) / np.linalg.norm(y-mean(y)) ) )
print(R2)
plt.legend()
plt.ylabel('alpha in the eigenvalues')
plt.xlabel('alpha in the degrees')
plt.show()



#Plot a histogram of the eigenvalues
plt.hist((my_array[0])[eig_idx], bins = 40)
plt.xlabel('alpha in the eigenvalues')
plt.ylabel('Count')
plt.show()

twopointfive = np.where(my_array[0] >= 2.5)[0]
idx1 = [val for val in twopointfive if val in eig_idx]
threepointfive = np.where(my_array[0] >= 3.5)[0]
idx2 = [val for val in threepointfive if val in eig_idx]
print(len(idx1))
print(len(idx2))

#Plot n vs largest eigenvalue
fig = plt.figure()
ax1 = fig.add_subplot(111)
x1 = my_array[4]
y1 = my_array[5]
ax1.scatter(x1,y1, c = 'b')
x2 = (my_array[4])[eig_idx]
y2 = (my_array[5])[eig_idx]
ax1.scatter(x2, y2, c = 'r',label = 'has power law fit in the eigenvalues')
A = np.vstack([x1,np.ones(len(x1))]).T
m,c = np.linalg.lstsq(A,y1)[0]
plt.plot(x1,m*x1+c, 'g', label = 'Fitted Line')
print('m = ',m)
print('c = ',c)

plt.legend()
plt.xlabel('size of the graph')
plt.ylabel('largest eigenvalue')
plt.show()


####AS_CLASS
eig_idx = np.where(as_array[1] >= 0.1)[0]
deg_idx = np.where(as_array[3] >= 0.1)[0]
idx = [val for val in eig_idx if val in deg_idx]
print('Total number of as-graphs analyzing: ', len(as_array[1]))
print('Number of power-laws in as eigs: ', len(eig_idx))
print('Number of power-laws in as degs: ', len(deg_idx))
print('Number of power-laws in as both: ', len(idx))


#Make a plot of the alpha-eig vs alpha-deg
x = (as_array[2])[idx] #degs
y = (as_array[0])[idx] #eigs

#Find a Least-Squares fit to the data
A = np.vstack([x,np.ones(len(x))]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y,label = 'Original Data')
plt.plot(x,m*x+c, 'r', label = 'Fitted Line')
print('m = ',m)
print('c = ',c)
p = np.vstack([m,c])
r = y - A.dot(p)
R2 = 1.0 - np.square( (np.linalg.norm(r) / np.linalg.norm(y-mean(y)) ) )
print(R2)
plt.legend()
plt.ylabel('alpha in the eigenvalues')
plt.xlabel('alpha in the degrees')
plt.show()



#Plot a histogram of the eigenvalues
plt.hist((as_array[0])[eig_idx], bins = 40)
plt.xlabel('alpha in the eigenvalues')
plt.ylabel('Count')
plt.show()

####OREGON_CLASS
eig_idx = np.where(oregon_array[1] >= 0.1)[0]
deg_idx = np.where(oregon_array[3] >= 0.1)[0]
idx = [val for val in eig_idx if val in deg_idx]
print('Total number of oregon-graphs analyzing: ', len(oregon_array[1]))
print('Number of power-laws in oregon eigs: ', len(eig_idx))
print('Number of power-laws in oregon degs: ', len(deg_idx))
print('Number of power-laws in oregon both: ', len(idx))


#Make a plot of the alpha-eig vs alpha-deg
x = (oregon_array[2])[idx] #degs
y = (oregon_array[0])[idx] #eigs

#Find a Least-Squares fit to the data
A = np.vstack([x,np.ones(len(x))]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y,label = 'Original Data')
plt.plot(x,m*x+c, 'r', label = 'Fitted Line')
print('m = ',m)
print('c = ',c)
p = np.vstack([m,c])
r = y - A.dot(p)
R2 = 1.0 - np.square( (np.linalg.norm(r) / np.linalg.norm(y-mean(y)) ) )
print(R2)
plt.legend()
plt.ylabel('alpha in the eigenvalues')
plt.xlabel('alpha in the degrees')
plt.show()
#Plot a histogram of the eigenvalues
plt.hist((oregon_array[0])[eig_idx], bins = 40)
plt.xlabel('alpha in the eigenvalues')
plt.ylabel('Count')
plt.show()




####MODEL_CLASS
eig_idx = np.where(model_array[1] >= 0.1)[0]
deg_idx = np.where(model_array[3] >= 0.1)[0]
idx = [val for val in eig_idx if val in deg_idx]
print('Total number of model-graphs analyzing: ', len(model_array[1]))
print('Number of power-laws in model eigs: ', len(eig_idx))
print('Number of power-laws in model degs: ', len(deg_idx))
print('Number of power-laws in model both: ', len(idx))


#Make a plot of the alpha-eig vs alpha-deg
x = (model_array[2])[idx] #degs
y = (model_array[0])[idx] #eigs

#Find a Least-Squares fit to the data
A = np.vstack([x,np.ones(len(x))]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y,label = 'Original Data')
plt.plot(x,m*x+c, 'r', label = 'Fitted Line')
print('m = ',m)
print('c = ',c)
p = np.vstack([m,c])
r = y - A.dot(p)
R2 = 1.0 - np.square( (np.linalg.norm(r) / np.linalg.norm(y-mean(y)) ) )
print(R2)
plt.legend()
plt.ylabel('alpha in the eigenvalues')
plt.xlabel('alpha in the degrees')
plt.show()
#Plot a histogram of the eigenvalues
plt.hist((model_array[0])[eig_idx], bins = 40)
plt.xlabel('alpha in the eigenvalues')
plt.ylabel('Count')
plt.show()



####RESAMPLE_CLASS
resample_graphs = np.where(my_array[6]> 0.0)[0]
resample_idx = np.where(my_array[7] >= 0.1)[0]
deg_idx = np.where(my_array[3] >= 0.1)[0]
eig_idx = np.where(my_array[1] >=0.1)[0]
idx = [val for val in eig_idx if val in resample_idx]
print('Total number of resample-graphs analyzing: ', len(resample_graphs))
print('Number of power-laws in as resampled eigs: ', len(resample_idx))
print('Number of power-laws in as reg eigs: ', len(eig_idx))
print('Number of power-laws in as both: ', len(idx))


#Make a plot of the alpha-eig vs alpha-deg
x = (my_array[0])[idx] #eigs
y = (my_array[6])[idx] #lap-eigs

#Find a Least-Squares fit to the data
#A = np.vstack([x,np.ones(len(x))]).T
#m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y,label = 'Original Data')
#plt.plot(x,m*x+c, 'r', label = 'Fitted Line')
#print('m = ',m)
#print('c = ',c)
#p = np.vstack([m,c])
#r = y - A.dot(p)
#R2 = 1.0 - np.square( (np.linalg.norm(r) / np.linalg.norm(y-mean(y)) ) )
#print(R2)
plt.legend()
plt.xlabel('alpha in the eigenvalues')
plt.ylabel('alpha in the resampled eigenvalues')
plt.show()



#Plot a histogram of the eigenvalues
plt.hist((my_array[6])[resample_idx], bins = 40)
plt.xlabel('alpha in the resampled eigenvalues')
plt.ylabel('Count')
plt.show()
