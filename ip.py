#!/usr/bin
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#import seaborn as sns
#plt.style.use('seaborn-white')
import pandas as pd
#import pandas_bokeh
import ast
from scipy.linalg import norm
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

current_path = os.getcwd()
img_path = os.path.join(current_path, "./static/coffe_table_cropped")
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.
from sklearn.cluster import MiniBatchKMeans
# import the necessary packages
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import sys
import argparse
#import  python_utils as utils
import cv2

# #############################################################################
# Defining the number of cluster for the colors
num_clusters=3
# Defining the number of cluster for the color composition clusters
threshold = 120
#Threshhold for the colro difference between two images
# #############################################################################
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist
    
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar
def find_link(image):
    with open("./link.txt",'a+') as f:
        while line != '':
            line = f.readline()
            if image in line: return line

def img_label(test_image, num_clusters_catalog=50):

    '''
    ##############################################################################
    X= []
    color_comp = open("./color_comp.txt",'w+')
    color_comp = open("./color_comp.txt",'a+')
    list_image = open("./list_image.txt",'w+')
    list_image = open("./list_image.txt",'a+')
    color_comp.write("Colors;Number of occurance\n")
    color_comp.close()
    n = 0
    list_img_n=[]
    for img in os.listdir(img_path):
        color_comp = open("./color_comp.txt",'a+')
        try:
            img = os.path.join(img_path,img)
            image = cv2.imread(img)
            image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print(image.shape)
            image = rgb2lab(np.uint8(np.asarray([image])))
            #I used rgb2lab to convert the values and then find the difference using deltaE_cie76
            # load the image and convert it from BGR to RGB so that
            # we can dispaly it with matplotlib
            # reshape the image to be a list of pixels
            image = image.reshape((image.shape[0]*image.shape[1] * image.shape[2], 3))
            # cluster the pixel intensities
            clt = KMeans(n_clusters = num_clusters)
            clt.fit(image)
            counts = Counter(clt.labels_)
            print(counts)
            counts_list = list(counts.values())
            counts_list.sort(reverse=True)
            keys_sorted = sorted(counts, key=counts.get, reverse=True)
            #counts_list.sort()
            occ_n = np.unique(clt.labels_, return_counts=True)[1]/sum(np.unique(clt.labels_, return_counts=True)[1])
            #ordered_colors = [clt.cluster_centers_[i] for i in sorted(counts, key=counts.get, reverse=True)]
            #print(ordered_colors[0])
            #print(clt.cluster_centers_)
            #color_comp.write("{} {}\n".format(clt.cluster_centers_[0], np.unique(clt.labels_, return_counts=True)[1]))
            ord_colors = [clt.cluster_centers_[i] for i in sorted(counts, key=counts.get, reverse=True)]
            #print(ord_colors)
            [color_comp.writelines(["%.6f " % ord_colors[j][i] for i in range(3)]) for j in range(num_clusters)]
            color_comp.writelines(["%0.6f " % counts_list[i] for i in range(num_clusters)])
            color_comp.writelines("\n")
            n += 1
            list_img_n.append([img, n-1])
        except:
            pass
    [list_image.write('{}\n'.format(l)) for l in list_img_n]

    list_image.close()
    color_comp.close()



    def Lab_color(image, image_colors, threshold = 60, number_of_colors = num_clusters):
        selected_color = rgb2lab(np.uint8(np.asarray([[image_colors])))
        select_image = False
        for i in range(number_of_colors):
            curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
            # diff = deltaE_cie76(selected_color, curr_color)
            #if (diff < threshold):
            #select_image = True
        return  curr_color
    '''




    import ast

    input= open("./color_comp.txt",'r')
    line = input
    X = []
    input.readline()

    def noL(input):
        i = 0
        while True:
            line = input.readline()
            i += 1
            if line == '':
                break
        return i

    nl = noL(input)
    input.close()
    input= open("./color_comp.txt",'r')
    input.readline()


    def parse(line):
        y=[]
        for x in line:
            x = ast.literal_eval(x)
            y.append(x)
        return y

    print(nl)
    for i in range(nl-1):
        line = input.readline()
        line = line.split()[0:3]
        parse(line)
        X.append(parse(line))
    input.close()

    #print(X)
    X = np.array(X)
    #print(X[0])
    #print(X)
    clt_cat = KMeans(n_clusters = num_clusters_catalog, random_state=0)
    clt_cat.fit(X)
    #print(clt_cat.cluster_centers_)
    #print(clt_cat.labels_)
    '''
    histohram plot cluster
    plt.hist(clt_cat.labels_,bins=num_clusters_catalog)
    plt.title('Histogram of Color Comp')
    plt.xlabel('Comp')
    plt.ylabel('Frequency')
    plt.savefig('clusters{}'.format(num_clusters_catalog))
    '''
    list_img_n =[]
    list_img_n= [line.strip('\n') for line in open("./list_image.txt",'r')]
    list_link =[]
    list_link= [line.strip('\n') for line in open("./Im_Link.txt",'r')]
    #print("list_link", list_link)
    # #############################################################################
    # using K-Mean to cluster color compositions
    img_path = os.path.join(current_path, "./static/test/cropped")
    print(img_path)
    img =  os.path.join(img_path,test_image)
    test = []
    test1 =[]
    test2= []
    img = os.path.join(img_path,img)
    image = cv2.imread(img)
    image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    image = rgb2lab(np.uint8(np.asarray([image])))
    #print(image.shape)
    #I used rgb2lab to convert the values and then find the difference using deltaE_cie76
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0]*image.shape[1] * image.shape[2], 3))
    clt = KMeans(n_clusters = num_clusters)
    clt.fit(image)
    counts = Counter(clt.labels_)
    print(counts)
    counts_list = list(counts.values())
    counts_list.sort(reverse=True)
    #print(counts_list)
    #counts_list = [(L*100)/(1.*sum(counts_list)) for L in counts_list]
    #counts_list.sort()
    occ_n = np.unique(clt.labels_, return_counts=True)[1]/sum(np.unique(clt.labels_,return_counts=True)[1])
    ord_colors = [clt.cluster_centers_[i] for i in sorted(counts, key=counts.get, reverse=True)]
    print(ord_colors)
    test1 = [ord_colors[j][i] for i in range(3) for j in range(num_clusters)]
    #print(test1)
    test2= counts_list
    #print(test2)
    test = test1[:][0:3]
    #print(test)
    l =[]
    n=0
    dn ={}
    list_n =[]
    #print(len(clt_cat.labels_))
    for i in clt_cat.labels_:
        n += 1
        if i == clt_cat.predict([test]):
            l.append(n)
    #print(l, n)
    dn = {l[i]: deltaE_cie76(X[l[i]],test) for i in range(len(l))}
    dn = sorted(dn.items(), key=lambda kv: kv[1])
    print("dn", dn ,"list", list_n)
    if dn[0][1] < threshold:
        list_n = [dn[i][0] for i in range(len(l)) if dn[i][1]<threshold]
        return [ast.literal_eval(list_img_n[i])[0] for i in list_n],[list_link[i] for i in list_n]
    else:
        print(num_clusters_catalog)
        num_clusters_catalog += 10
        img_label(test_image, num_clusters_catalog)


print(img_label('celestia-loveseat.jpg'))



'''


# #############################################################################
# Compute DBSCANfrom sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
def db_image_cluster(X):
    db = DBSCAN(eps=0.1, min_samples=5).fit(X)
    labels = db.labels_
    ncore_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    ncore_samples_mask[db.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels,
                                               average_method='arithmetic'))
    
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(image, labels))

db_image_cluster(X)

    #print([i for i, e in enumerate(clt_cat.labels_) if e ==clt_cat.predict([test])[0]])
    #print([list_img_n[0] for list_img_n[1] =
    #plt.show(
    #color_comp.writelines(["%.6f " % clt.cluster_centers_[0][i] for i in range(num_clusters)])
    #color_comp.writelines(["%d " % occ_n[i] for i in range(num_clusters)])
    #color_comp.writelines("\n")
    #except:
    #pass

#print(img_label('t1.jpg'))
#clt.predict(X[, sample_weight])

'''
