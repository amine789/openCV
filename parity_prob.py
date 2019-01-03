# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:29:44 2018

@author: amine bahlouli
"""

import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import shuffle

def init_weigth(M1,M2):
    return np.random.rand(M1,M2)*np.sqrt(2/M1)

def all_parity_pairs(nbit):
    N = 2**nbit
    remainder = 100 -(N%100)
    Ntotal = N+remainder
    X = np.zeros((Ntotal,nbit))
    Y = np.zeros(Ntotal)
    for ii  in range(Ntotal):
        i = ii % N
        for j in range(Ntotal):
            if i % (2**(j+1)) !=0:
                i-=2**j
                X[ii,j]=1
            Y[ii]=X[ii].sum()%2
    return X,Y


        

class simpleRNN:
    def __init__(self,M):
        self.M=M
        
    def fit(self,X,Y, learning_rate=10e-1,mu=0.99, activation=T.tanh, epochs=15):
        D = X.shape[0]
        K = len(set(Y.flatten()))
        N = len(Y)
    
        M=self.M
        self.f=activation
        
        Wx = init_weigth(D,M)
        Wh = init_weigth(M,M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        W0 = init_weigth(M,K)
        b0= np.zeros(K)
        
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.W0 = theano.shared(W0)
        self.b0 = theano.shared(b0)
        
        thX = T.fmatrix("X")
        thY = T.ivector("Y")
        
        def recurrence(x_t, h_t1):
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.W0) + self.b0)
            return h_t,y_t
        [h,y], _ = theano.scan(
                fn=recurrence,
                outputs_info= [self.h0,None],
                sequences = thX,
                n_steps = thX.shape[0]
                )
        py_x = y[:,0,:]
        prediction = T.argmax(py_x,axis=1)
        cost = -T.mean(T.log(py_x(T.arange(thY.shape[0], thY))))
        grad = T.grads(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        
        updates = [
                (p,p+mu*dp - learning_rate*g) for p,dp,g in zip(self.params,dparams,grad)]
        + [(dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grad)]
        
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function( inputs=[thX, thY], outputs=[cost, prediction], updates=updates,)
        
        costs = []
        
        for i in range(epochs):
            n_correct=0
            X,Y = shuffle(X,Y)
            for j in range(N):
                c,p,rout = self.train_op(X[j], X[j])
                if p[-1]==Y[j,-1]:
                    n_correct +=1
                print("y_shape: ", rout.shape)
                print("i: ",i,"cost: ",cost,"classification rate:",(float(n_correct)))
        