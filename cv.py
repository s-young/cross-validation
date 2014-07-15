# usage: python cv.py --genofile 'myAveImpGenotype_wheat183.csv' --phenofile 'y8new.txt' [--CVfolds 5 --ridge 0 -3 -9]

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from argparse import ArgumentParser

def parse_args():
    'Parse the command line arguments for the program.'
    parser = ArgumentParser(
        description="Cross validation for csv and txt files")
    parser.add_argument('--CVfolds', type=int,
        help='number of cv folds', nargs='+', default=3)
    parser.add_argument('--genofile', required=True, type=str,
        help='Input Genotype File')
    parser.add_argument('--phenfile', required=True, type=str,
       help='Input Phenotype File')
    parser.add_argument('--ridge', type=int, default=[0, -5],
        nargs='+', help='ridge parameter')
    return parser.parse_args()

def get_genotype(gen_filename):
    'Read the input genotype file.'
    genotype = np.genfromtxt(gen_filename, delimiter=' ')
    print(gen_filename)
    #print(genotype)
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
    a = 10 ** np.array(ridgepara)
    Imat = a * np.mat(np.eye(len(xtrain)))
    xt = np.transpose(xtrain)
    xxt = np.matrix(xtrain)*np.matrix(xt)
    w = (xxt+Imat).I
    b= np.matrix(xt)* np.matrix(w) *np.matrix(ytrain).T
    ypred = np.matrix(xtest)*b
    return ypred

def crossval(k,xtrain,ytrain, ridgepara):
    ' K folds cross validation.'
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
    return optpara   

def main():
    args = parse_args()
    k = args.CVfolds
    x = get_genotype(args.genofile)
    #genotype = np.genfromtxt('myAveImpGenotype_wheat183.csv', delimiter=' ')
    #x = np.transpose(genotype)
    print(x)
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
    optpara = crossval(k,xtrain,ytrain, args.ridge)
    ypred = rr(xtrain, ytrain,xtest, optpara)
    cor = np.corrcoef(np.matrix(ytest), np.transpose(ypred))
    corr_pred = cor[1,0]
    print('The optimal parameter by CV is %f' + '\n') % 10**optpara
    print('Pearson''s correlation btw true and pred is %f') % corr_pred

if __name__ == '__main__':
    main()
    
