'''
Created on 2018/11/30

@author: 0145215059
'''
#coding:utf-8
import time
import math
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import argparse

def distance(r1, r2):
    return math.sqrt((r1[1]-r2[1])**2 + (r1[0]-r2[0])**2)

class QMC :
    def __init__(self,n,p,m,l,b,r):
        self.TROTTER_DIM = n
        self.ANN_PARA = p
        self.ANN_STEP = m
        self.MC_STEP = l
        self.BETA = b
        self.reduc_para = r
        self.POINT = list()
        self.NCITY = 0
        self.TOTAL_TIME = 0

    def read(self,file):
        f = open(os.path.dirname(os.path.abspath(file))+"/"+file).read().split("\n")

        for i in f:
            self.POINT.append(i.split(" "))

        self.NCITY = len(self.POINT)
        self.TOTAL_TIME = self.NCITY
        for i in range(self.NCITY):
            self.POINT[i].remove(self.POINT[i][0])
        for i in range(self.NCITY):
            for j in range(2):
                 self.POINT[i][j] = float(self.POINT[i][j])

    # spin configurations at a time and some trotter dimension
    def spin_config_at_a_time_in_a_TROTTER_DIM(self,tag):
        config = list(-np.ones(self.NCITY, dtype = np.int))
        config[tag] = 1
        return config

    # sping configurations at some trotter dimension
    def spin_config_in_a_TROTTER_DIM(self,tag):
        spin = list()
        spin.append(config_at_init_time)
        for i in xrange(self.TOTAL_TIME-1):
            spin.append(list(self.spin_config_at_a_time_in_a_TROTTER_DIM(tag[i])))
        return spin

    # whole spin configurations
    def getSpinConfig(self):
        spin = list()
        for i in xrange(self.TROTTER_DIM):
            tag = np.arange(1,self.NCITY)
            np.random.shuffle(tag)
            spin.append(self.spin_config_in_a_TROTTER_DIM(tag))
        return spin

    def getBestRoute(self,config):   
        length = list()
        for i in xrange(self.TROTTER_DIM):
            route = list()
            for j in xrange(self.TOTAL_TIME):
                 route.append(config[i][j].index(1))
            length.append(self.getTotaldistance(route))

        min_Tro_dim = np.argmin(length)
        Best_Route = list()
        for i in xrange(self.TOTAL_TIME):
            Best_Route.append(config[min_Tro_dim][i].index(1))
        return Best_Route

    def getTotaldistance(self,route):
        Total_distance = 0
        for i in xrange(self.TOTAL_TIME):
            Total_distance += distance(self.POINT[route[i]],self.POINT[route[(i+1)%self.TOTAL_TIME]])/max_distance
        return Total_distance

    def getRealTotaldistance(self,route):
        Total_distance = 0
        for i in xrange(self.TOTAL_TIME):
            Total_distance += distance(self.POINT[route[i]], self.POINT[route[(i+1)%self.TOTAL_TIME]])
        return Total_distance

    def move(self,config):
        c = np.random.randint(0,self.TROTTER_DIM)
        a_ = range(1,self.TOTAL_TIME)
        a = np.random.choice(a_)
        a_.remove(a)
        b = np.random.choice(a_)
            
        p = config[c][a].index(1)
        q = config[c][b].index(1)

        delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0
                
        for j in range(self.NCITY):
            l_p_j = distance(self.POINT[p], self.POINT[j])/max_distance
            l_q_j = distance(self.POINT[q], self.POINT[j])/max_distance
            delta_costc += 2*(-l_p_j*config[c][a][p] - l_q_j*config[c][a][q])*(config[c][a-1][j]+config[c][(a+1)%self.TOTAL_TIME][j]) \
                 +2*(-l_p_j*config[c][b][p] - l_q_j*config[c][b][q])*(config[c][b-1][j]+config[c][(b+1)%self.TOTAL_TIME][j])

        para = (1/self.BETA)*math.log(math.cosh(self.BETA*self.ANN_PARA/self.TROTTER_DIM)/math.sinh(self.BETA*self.ANN_PARA/self.TROTTER_DIM))
        delta_costq_1 = config[c][a][p]*(config[(c-1)%self.TROTTER_DIM][a][p]+config[(c+1)%self.TROTTER_DIM][a][p])
        delta_costq_2 = config[c][a][q]*(config[(c-1)%self.TROTTER_DIM][a][q]+config[(c+1)%self.TROTTER_DIM][a][q])
        delta_costq_3 = config[c][b][p]*(config[(c-1)%self.TROTTER_DIM][b][p]+config[(c+1)%self.TROTTER_DIM][b][p])
        delta_costq_4 = config[c][b][q]*(config[(c-1)%self.TROTTER_DIM][b][q]+config[(c+1)%self.TROTTER_DIM][b][q])

        delta_cost = delta_costc/self.TROTTER_DIM+para*(delta_costq_1+delta_costq_2+delta_costq_3+delta_costq_4)
                     
        # HMC ??
        if delta_cost <= 0:
            config[c][a][p]*=-1
            config[c][a][q]*=-1
            config[c][b][p]*=-1
            config[c][b][q]*=-1
        elif np.random.random() < np.exp(-self.BETA*delta_cost):
            config[c][a][p]*=-1
            config[c][a][q]*=-1
            config[c][b][p]*=-1
            config[c][b][q]*=-1
                     
        return config
                     
#QMC simulation
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file",required=True,help="specify a file which contains cities's information.")
    parser.add_argument("--trotter_dim",type=int,default=10)
    parser.add_argument("--ann_para",type=float,default=1.0,help="initial annealing parameter")
    parser.add_argument("--ann_step",type=int,default=330)
    parser.add_argument("--mc_step",type=int,default=13320)
    parser.add_argument("--beta",type=float,default=float(37))
    parser.add_argument("--reduc_para",type=float,default=0.99)
    args = parser.parse_args()

    # prepare annealer
    anneal = QMC(args.trotter_dim,args.ann_para,args.ann_step,args.mc_step,args.beta,args.reduc_para)
    anneal.read(args.file)
    
    max_distance = 0
    for i in range(anneal.NCITY):
        for j in range(anneal.NCITY):
            if max_distance <= distance(anneal.POINT[i], anneal.POINT[j]):
                max_distance = distance(anneal.POINT[i], anneal.POINT[j])

    config_at_init_time = list(-np.ones(anneal.NCITY,dtype=np.int))
    config_at_init_time[0] = 1

    print "start..."
    t0 = time.clock()

    np.random.seed(100)
    spin = anneal.getSpinConfig()
    LengthList = list()
    for t in range(anneal.ANN_STEP):
        for i in range(anneal.MC_STEP):
            con = anneal.move(spin)
            rou = anneal.getBestRoute(con)
            length = anneal.getRealTotaldistance(rou)
        LengthList.append(length)
        print "Step:{}, Annealing Parameter:{}, length:{}".format(t+1,anneal.ANN_PARA, length)
        anneal.ANN_PARA *= anneal.reduc_para

    Route = anneal.getBestRoute(spin)
    Total_Length = anneal.getRealTotaldistance(Route)
    elapsed_time = time.clock()-t0

    print "shortest path is {}".format(Route)
    print "shortest lenght is {}".format(Total_Length)
    print "processing time is {}s".format(elapsed_time)

    plt.plot(LengthList)
    plt.show()

