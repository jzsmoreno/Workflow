import pandas as pd
from numba import njit, prange, jit
import numpy as np
import random

@jit()
def centros(X,evalu,n_days):
    C=[]
    aux=random.choice(X)
    while(aux[1]!=1):
        aux=random.choice(X)
    C.append(int(aux[0]))
    dist=evalu[int(C[0]-1)]
    aux=np.random.choice(np.array(dist))
    while(aux<=np.mean(np.array(dist))*0.7):
        aux=np.random.choice(np.array(dist))
    indx=dist.index(aux)
    C.append(int(X[indx][0]))
    for i in range(n_days-2):
        dist=np.array(dist)
        dist+=np.array(evalu[int(C[i+1]-1)])
        dist=list(dist)
        aux=np.random.choice(np.array(dist))
        while(aux<=np.mean(np.array(dist))*0.9):
            aux=np.random.choice(np.array(dist))
        indx=dist.index(aux)
        C.append(int(X[indx][0]))    
    return C

def clientes(C,X,asig,frec,data,evalu,n_days):
    c1 = int(1*10**9)
    c2 = c1*10
    ntotal=len(X)
    climit=data.Frecuencia.sum()/n_days
    vmax=data.Frecuencia*data.Vol_Entrega
    vlim=vmax.sum()/n_days
    for i in prange(n_days):
        dist=list(evalu[C[i]-1].copy())
        dist[C[i]-1]= c1
        asig[i][C[i]-1]=1
        frec[C[i]-1]-=1
        maxv=X[C[i]-1][2]
        j=1
        f=[]
        for k in prange(len(frec)):
            if(frec[k]>1):
                f.append(dist[k])
        for k in range(len(f)):
            aux=min(f)
            ind=f.index(aux)
            indx=dist.index(aux)
            while(frec[indx]<1):
                dist[indx]= c2
                f[ind]= c2
                aux=min(f)
                indx=dist.index(aux)
                ind=f.index(aux)
            dist[indx]= c1
            f[ind]= c2
            asig[i][indx]=1
            frec[indx]-=1
            maxv+=X[indx][2]
            j=j+1
        while(sum(frec)>=1):
                aux=min(dist)
                indx=dist.index(aux)
                while(frec[indx]<1):
                    dist[indx]= c2
                    aux=min(dist)
                    indx=dist.index(aux)
                dist[indx]= c1
                asig[i][indx]=1
                frec[indx]-=1
                maxv+=X[indx][2]
                j=j+1
                if(maxv>=vlim*1.1):
                    break
                if(j>=climit*1.1):
                    break
        #print("Volumen Total del Día ",i,": ",maxv,"Número Total de clientes: ",j)
        
def grasp(data,evalu,n_days):
    X=data[['Id_Cliente',"Frecuencia","Vol_Entrega","lat","lon"]].to_numpy()
    ntotal=len(X)
    X=list(X)
    k=False
    while( k==False):
        asig=[]
        for i in prange(n_days):
            ss=[]
            for j in prange(ntotal):
                ss.append(0)
            asig.append(ss)
        frec=list(np.array(data.Frecuencia))

        a=centros(X,evalu,n_days)

        clientes(a,X,asig,frec,data,evalu,n_days)
        j=0

        for i in frec:
            if(i==0):
                j+=1
        if(j==3625):
            k=True
    return asig