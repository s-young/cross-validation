# usage: python cv.py --genofile 'myAveImpGenotype_wheat183.csv' --phenfile 'y8new.txt' [--CVfolds 5 --ridge 0 -3 -9]

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from argparse import ArgumentParser
from collections import Counter

def parse_args():
    'Parse the command line arguments for the program.'
    parser = ArgumentParser(
        description="Cross validation for csv and txt files")
    parser.add_argument('--CVfolds', type=int,
        help='number of cv folds', default=3)
    parser.add_argument('--genofile', required=True, type=str,
        help='Input Genotype File')
    parser.add_argument('--phenfile', required=True, type=str,
       help='Input Phenotype File')
    parser.add_argument('--ridge', type=int, default=range(-3,3),
       help='ridge parameters') 
    parser.add_argument('--it', type =int, default = 50,
       help = 'number of iterations')
    return parser.parse_args()

def get_genotype(gen_filename):
    'Read the input genotype file.'
    genotype = np.genfromtxt(gen_filename, delimiter=' ')
    print(gen_filename)
    x = np.transpose(genotype)
    return x

def get_phenotype(phen_filename):
    'Read the input phenotype file.'
    phen = open(phen_filename)
    y = []
    for line in phen:
	line = line.strip()
	y.append(float(line))
    y = np.transpose(y)
    return y 

def rr(xtrain, ytrain, xtest, ridgepara):
    'Ridge regression program for a given ridge parameter.'
    a = 10 ** np.array(ridgepara)
    Imat = a * np.mat(np.eye(len(xtrain)))
    xt = np.transpose(xtrain)
    xxt = np.matrix(xtrain)*np.matrix(xt)
    w = (xxt+Imat).I
    b= np.matrix(xt)* np.matrix(w) *np.matrix(ytrain).T
    ypred = np.matrix(xtest)*b
    return ypred

def crossval(k,xtrain,ytrain, ridgepara):
    'K folds cross validation for a set of given parameters.'
    n = len(xtrain)
    indices = np.arange(n)
    p = len(ridgepara)
    corr_output = np.zeros(shape=(k,p)) # = np.random.random(k,p)
  
    for j in np.arange(len(ridgepara)):
	newid = indices
       	for fold in range(k):
	  tstID = newid[0:n/k]
	  tstacc = tstID # Accumulated indices of test set
	  trnID = newid[n/k:]
	  trnrmn = trnID # Remaining indices of train set
	  pred = rr(xtrain[trnID,:], ytrain[trnID],xtrain[tstID,:], ridgepara[j])
	  corr = np.corrcoef(np.matrix(ytrain[tstID]), np.transpose(pred))
	  corr_output[fold,j] = corr[1,0]
	  newid = np.hstack((indices[trnrmn],indices[tstacc]))

    corr_sum = map(sum,zip(*corr_output))
    optpara = ridgepara[corr_sum.index(max(corr_sum))]
    # print('The opt para is %f') % 10**optpara
    return optpara   

def main():
    args = parse_args()
    k = args.CVfolds
    x = get_genotype(args.genofile)
    m = len(x)
    print(x.shape)
    y = get_phenotype(args.phenfile)
    whole = random.sample(np.arange(m), m)
    trainid = whole[0:m*3/4]
    testid = whole[m*3/4:]
    xtrain = x[trainid, :]
    xtest = x[testid, :]
    ytrain = y[trainid]
    ytest = y[testid]
    paras = args.ridge

    Iter = args.it
    predictions = []
    optparameters = []
    correlations = []
    for iteration in range(Iter):
        optpara = crossval(k,xtrain,ytrain, args.ridge)
	optparameters.append(optpara)
        ypred = rr(xtrain, ytrain,xtest, optpara)
        predictions.append(ypred)
        cor = np.corrcoef(np.matrix(ytest), np.transpose(ypred))
        corr_pred = cor[1,0]
        correlations.append(corr_pred)
    lens =[]
    for p in range(len(paras)):
        freqofpara = optparameters.count(paras[p])
        lens.append(freqofpara)
    opt = paras[lens.index(max(lens))]
    print('The optimal parameter by CV is %f' + '\n') % 10**opt
    print('Pearson''s correlation btw true and pred is %f') % np.mean(correlations)

if __name__ == '__main__':
    main()
    
