'''
library for quantum annealing

(C) Yong-Gwi Cho, Sony LSI Design Inc.
'''
import random
import numpy as np
import os
import math
from datetime import datetime        


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
        self.max_distance = 0

    def read(self,file):
        f = open(file).read().split("\n")

        for i in f:
             if (i.split(" ")[0]).isdigit() : # ignore header of data
                self.POINT.append(i.split(" "))

        self.NCITY = len(self.POINT)
        self.TOTAL_TIME = self.NCITY
        for i in range(self.NCITY):
            self.POINT[i].remove(self.POINT[i][0])
        for i in range(self.NCITY):
            for j in range(2):
                 self.POINT[i][j] = float(self.POINT[i][j])

    # spin configurations at a time and some trotter dimension
    def spin_conf_at_a_time_in_a_TROTTER_DIM(self,tag):
        conf = list(-np.ones(self.NCITY, dtype = np.int))
        conf[tag] = 1
        return conf

    # sping configurations at some trotter dimension
    def spin_conf_in_a_TROTTER_DIM(self,tag,conf):
        spin = list()
        spin.append(conf)
        for i in xrange(self.TOTAL_TIME-1):
            spin.append(list(self.spin_conf_at_a_time_in_a_TROTTER_DIM(tag[i])))
        return spin

    # whole spin configurations
    def getSpinConf(self,conf):
        spin = list()
        for i in xrange(self.TROTTER_DIM):
            tag = np.arange(1,self.NCITY)
            np.random.shuffle(tag)
            spin.append(self.spin_conf_in_a_TROTTER_DIM(tag,conf))
        return spin

    def getBestPath(self,conf):
        length = list()
        for i in xrange(self.TROTTER_DIM):
            path = list()
            for j in xrange(self.TOTAL_TIME):
                 path.append(conf[i][j].index(1))
            length.append(self.getTotaldistance(path))

        min_Tro_dim = np.argmin(length)
        Best_Path = list()
        for i in xrange(self.TOTAL_TIME):
            Best_Path.append(conf[min_Tro_dim][i].index(1))
        return Best_Path

    def getTotaldistance(self,route):
        Total_distance = 0
        for i in xrange(self.TOTAL_TIME):
            Total_distance += distance(self.POINT[route[i]],self.POINT[route[(i+1)%self.TOTAL_TIME]])/self.max_distance
        return Total_distance

    def getRealTotaldistance(self,route):
        Total_distance = 0
        for i in xrange(self.TOTAL_TIME):
            Total_distance += distance(self.POINT[route[i]], self.POINT[route[(i+1)%self.TOTAL_TIME]])
        return Total_distance

    def save(self):
        file = open("./spin_conf-"+datetime.now().isoformat(),mode=w)
        for k in range():
            for j in range():
                for i in range():
                    file.write(conf[i][j][k]+"\n")
        file.close()
        return 0
    
    def move(self,conf):
        c = np.random.randint(0,self.TROTTER_DIM)
        a_ = range(1,self.TOTAL_TIME)
        a = np.random.choice(a_)
        a_.remove(a)
        b = np.random.choice(a_)
            
        p = conf[c][a].index(1)
        q = conf[c][b].index(1)

        delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0
                
        for j in range(self.NCITY):
            l_p_j = distance(self.POINT[p], self.POINT[j])/self.max_distance
            l_q_j = distance(self.POINT[q], self.POINT[j])/self.max_distance
            delta_costc += 2*(-l_p_j*conf[c][a][p] - l_q_j*conf[c][a][q])*(conf[c][a-1][j]+conf[c][(a+1)%self.TOTAL_TIME][j]) \
                           +2*(-l_p_j*conf[c][b][p] - l_q_j*conf[c][b][q])*(conf[c][b-1][j]+conf[c][(b+1)%self.TOTAL_TIME][j])

        para = (1/self.BETA)*math.log(math.cosh(self.BETA*self.ANN_PARA/self.TROTTER_DIM)/math.sinh(self.BETA*self.ANN_PARA/self.TROTTER_DIM))
        delta_costq_1 = conf[c][a][p]*(conf[(c-1)%self.TROTTER_DIM][a][p]+conf[(c+1)%self.TROTTER_DIM][a][p])
        delta_costq_2 = conf[c][a][q]*(conf[(c-1)%self.TROTTER_DIM][a][q]+conf[(c+1)%self.TROTTER_DIM][a][q])
        delta_costq_3 = conf[c][b][p]*(conf[(c-1)%self.TROTTER_DIM][b][p]+conf[(c+1)%self.TROTTER_DIM][b][p])
        delta_costq_4 = conf[c][b][q]*(conf[(c-1)%self.TROTTER_DIM][b][q]+conf[(c+1)%self.TROTTER_DIM][b][q])

        delta_cost = delta_costc/self.TROTTER_DIM+para*(delta_costq_1+delta_costq_2+delta_costq_3+delta_costq_4)
                     
        # accep spin flippign by min(1,exp(-beta*delta_cost))
        if delta_cost <= 0:
            conf[c][a][p]*=-1
            conf[c][a][q]*=-1
            conf[c][b][p]*=-1
            conf[c][b][q]*=-1
        elif np.random.random() < np.exp(-self.BETA*delta_cost):
            conf[c][a][p]*=-1
            conf[c][a][q]*=-1
            conf[c][b][p]*=-1
            conf[c][b][q]*=-1
                     
        return conf
