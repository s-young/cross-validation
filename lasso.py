# usage: python lassotry.py --genofile 'myAveImpGenotype_wheat183.csv' --phenfile 'y8new.txt' {--CVfolds 5 --lasso [0.5, 5]}

from sklearn import linear_model
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
    parser.add_argument('--lasso', type=int, default=[1.25, 1.5, 1.75, 2, 2.5, 3.5],
        help='lasso parameters') 
    parser.add_argument('--it', type =int, default = 100,
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

def lassocvpred(xtrain, ytrain, xtest, alphas):  
# alternatively, lassocvpara(xtrain, ytrain, alphas): ...  return optpara
    'Lasso regression for a given set of lasso parameters.'
    
    model = linear_model.LassoCV(cv=3, alphas=alphas,fit_intercept=False, tol=0.01).fit(xtrain, ytrain)
    optpara = model.alpha_
    coefs = model.coef_
    optcoef = np.asmatrix(coefs)
    predcv = np.matrix(xtest)*np.transpose(optcoef)
    return (optpara, predcv, coefs) 

def main():
    args = parse_args()
    alphas = args.lasso
    k = args.CVfolds
    x = get_genotype(args.genofile)
    m = len(x)
    print(x.shape)
    y = get_phenotype(args.phenfile)
    Iter = args.it
    paras = args.lasso
    predictions = []
    optparameters = []
    correlations = []
    print(":-))) %i iterations") % Iter
    for iteration in range(Iter):
        whole = random.sample(np.arange(m), m)
        trainid = whole[0:m*3/4]
        testid = whole[m*3/4:]
        xtrain = x[trainid, :]
        xtest = x[testid, :]
        ytrain = y[trainid]
        ytest = y[testid]
        
        optpara, ypred,coefs = lassocvpred(xtrain,ytrain, xtest, paras)
	cor = np.corrcoef(np.matrix(ytest), np.transpose(ypred))
        corr_pred = cor[1,0]
        correlations.append(corr_pred)
        optparameters.append(optpara)
    print('The candidate parameters are %s') % paras
    print('The optparas in each iteration are %s') % optparameters
 #   print('The correlations in each iteration are %s') % correlations

    freqofpara=[]
    for p in range(len(paras)):
        freqofpara.append (optparameters.count(paras[p]))
    itra = float(Iter)
    freq = [x/itra for x in freqofpara]
    opt = paras[freqofpara.index(max(freqofpara))]
    print('The frequency that each optpara appears is %s') % freq

    print('OutputI: The most frequently selected optpara by CVs is %f' ) % opt
    subcorr =[]
    for q in range(len(correlations)):
        if optparameters[q] == opt:
           subcorr.append(correlations[q])
    print('OutputI: The corresponding Pearson''s correlation is %f'+ '\n') % np.mean(subcorr)

    opt2 = [a*b for a,b in zip(freq, paras)]
    optpara = sum(opt2)
    print('OutputII: The weighted optimal parameter by CVs is %f' ) % optpara
    freqofcorr = []
    for i in range(len(paras)):
       for j in range(Iter):
           if paras[i]==optparameters[j]:
              freqofcorr.append(correlations[j])
    cor2 =[c*d for c,d in zip(freq, freqofcorr)]
    optcor = sum(cor2)
    print('OutputII: Weighted Pearson''s correlation btw true and pred is %f') % optcor
if __name__ == '__main__':
    main()
    

