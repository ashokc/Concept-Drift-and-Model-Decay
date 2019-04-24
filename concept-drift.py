import sys

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

import random as rn
import numpy as np
import matplotlib.pyplot as plt
rn.seed(1)
np.random.seed(1)

#   f(x,y) = w[0] + w[1] * x + w[2] * y
#   prp ./concept.py 10000

def generateData(w, n):
    x = (lmax - lmin) * np.random.random_sample(n*10) + lmin
    y = (lmax - lmin) * np.random.random_sample(n*10) + lmin
    lineY = - ( w[0] + w[1] * x ) / w[2]    # Decision Boundary: w[0] + w[1] * x + w[2] * y
    fvalsY = y - lineY
    lineX = - ( w[0] + w[2] * y ) / w[1]
    fvalsX = x - lineX
    data, labels = [], []
    zeroCount, oneCount = 0, 0
    for i, fy in enumerate(fvalsY):
        fx = fvalsX[i]
        if (zeroCount < n) and (fx < -1.0e-1):
            labels.append(0.0)
            data.append([x[i],y[i]])
            zeroCount = zeroCount + 1
        if (oneCount < n) and (fx > 1.0e-1):
            labels.append(1.0)
            data.append([x[i],y[i]])
            oneCount = oneCount + 1
        if (zeroCount == n) and (oneCount == n):
            break;
    data=np.array([np.array(xi) for xi in data])
    labels = np.array(labels)
    return data, labels

def relabelData (x, y, w):
    lineY = - ( w[0] + w[1] * x ) / w[2]    # Current Decision Boundary: w[0] + w[1] * x + w[2] * y
    fvalsY = y - lineY
    lineX = - ( w[0] + w[2] * y ) / w[1]
    fvalsX = x - lineX
    labels = []
    for i, fy in enumerate(fvalsY):
        fx = fvalsX[i]
        if (fx < 0.0):
            labels.append(0.0)
        if (fx > 0.0):
            labels.append(1.0)
    labels = np.array(labels)
    return labels

def plotData(w0, w, data, labels, marker, angle_in, f_score_in):
    angle = str(round(angle_in, 2))
    f_score = str(round(f_score_in, 3))
    fig = plt.figure(figsize=(8,8),dpi=720)
    data0 = data[np.where(labels == 0.0)[0]]
    data1 = data[np.where(labels == 1.0)[0]]
    plt.xlim(lmin-0.05, lmax+0.05)
    plt.ylim(lmin-0.05, lmax+0.05)
    xvals = np.linspace(lmin-0.05, lmax+0.05, 100)
    line_data0 = np.stack((xvals, ( - w0[0] - w0[1] * xvals ) / w0[2]),axis=-1)
    plt.plot(line_data0[:, 0], line_data0[:, 1], color='b', linewidth=2)
    line_data = np.stack((xvals, ( - w[0] - w[1] * xvals ) / w[2]),axis=-1)
    plt.plot(line_data[:, 0], line_data[:, 1], color='k',linewidth=2)
    plt.plot(data0[:, 0], data0[:, 1], color='r', linestyle='', markersize=1,marker='.')
    plt.plot(data1[:, 0], data1[:, 1], color='g', linestyle='', markersize=1,marker='.')
    fig.tight_layout()
    fig.savefig('data-plot-' + marker + '-' + angle + '-' + f_score + '.png', format='png', dpi=720)
    plt.close()

def getF1Score (dataIn, labelsIn):
    predictedLabels = model.predict(dataIn)
    c_report = classification_report(labelsIn, predictedLabels, digits=4, target_names=target_names, output_dict=True)
    return c_report['weighted avg']['f1-score']

def buildModel(data, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(data, labels)
    train_indices, test_indices = next(sss)
    train_data, test_data  = data[train_indices], data[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    model.fit(train_data, train_labels)
    return test_data, test_labels

def sampleAndPredict(k,theta0, w_in):
    count, delTheta, theta = 0, 0.2, theta0
    new_f1_score = current_f1_score
    w = w_in.copy()
    threshold_f1_score = 0.925
    while (new_f1_score > threshold_f1_score):
        count = count + 1
        theta = theta + delTheta
        w[2] = - w[1] / np.tan(theta*np.pi/180.0)
        new_data, new_labels = generateData(w, nNew)
        sampled = np.random.randint(0, nNew, size=nSample)
        if (count == 1):
            sampledData = new_data[sampled]
            sampledLabels = new_labels[sampled]
            all_new_data = new_data.copy()
        else:
            sampledData = np.vstack( (sampledData, new_data[sampled]) )
            sampledLabels = np.concatenate( (sampledLabels, new_labels[sampled]), axis=None )
            all_new_data = np.vstack( (all_new_data, new_data) )
    
        new_f1_score = getF1Score(sampledData, sampledLabels)
        print (theta, new_f1_score)
        thetas.append(theta)
        f1_scores.append(new_f1_score)
    plotData(w0, w, sampledData, sampledLabels, str(k) + '-sampled', theta, new_f1_score)
    return all_new_data, new_f1_score, w

args = sys.argv
nPoints = int(args[1])
target_names = ['zero', 'one']
lmin, lmax = -0.5, 0.5

nNew = int(nPoints/8)
#nSample = max(5,min(50, int(nNew/10)))
nSample = 25
f1_scores, thetas = [], []
w0 = [0.0, np.tan(np.pi*5/180), -1.0]
w = w0.copy()
data, labels = generateData(w, nPoints)
model = LinearSVC(tol=1.0e-6,max_iter=20000,verbose=0)
test_data, test_labels = buildModel(data, labels)
current_f1_score = getF1Score(test_data, test_labels)
w = [model.intercept_[0], model.coef_[0][0], model.coef_[0][1] ]
theta0 = np.arctan(-model.coef_[0][0] / model.coef_[0][1]) * 180.0/np.pi
plotData(w0, w, data, labels, '0-all', theta0, current_f1_score)
print ('i/f1-score/theta0/w', 0,current_f1_score,theta0,w)
f1_scores.append(current_f1_score)
thetas.append(theta0)
for i in range(0, 4):
    new_data, current_f1_score, current_w = sampleAndPredict(i,theta0, w)
    w0 = current_w.copy()
    w = current_w.copy()
    data = np.vstack( (data, new_data) )
    labels = relabelData (data[:,0], data[:, 1], w)
    test_data, test_labels = buildModel(data, labels)
    current_f1_score = getF1Score(test_data, test_labels)
    w = [model.intercept_[0], model.coef_[0][0], model.coef_[0][1] ]
    theta0 = np.arctan(-model.coef_[0][0] / model.coef_[0][1]) * 180.0/np.pi
    if (theta0 < 0):
        theta0 = 180.0 + theta0
    thetas.append(theta0)
    f1_scores.append(current_f1_score)
    print ('i/f1-score/theta0/w', i,current_f1_score,theta0,w)
    plotData(w0, w, data, labels, str(i) + '-all', theta0, current_f1_score)

#plotData(w0, w, data, labels, i+1, int(theta0))

#    data = np.vstack( (data, new_sampledData) )
#    labels = np.concatenate( (labels, new_sampledLabels), axis=None )

fig = plt.figure(figsize=(8,8),dpi=720)
plt.plot(thetas, f1_scores)
fig.savefig('theta-f1-score.png', format='png', dpi=720)
fig.tight_layout()
plt.close()

