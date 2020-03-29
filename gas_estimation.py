# -*- coding: utf-8 -*-
"""
@author: alexandrugiurca
"""

### imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import scipy.stats as st
import scipy.special as scs
from pandas.plotting import scatter_matrix
import grad as gd
from grad import * 


def fnPlotData(sX,lY,df,sXlabel,sYlabel,sName,vMean):
    """
    Purpose:
        Plot dataframe with more values regarding x value and their mean 
   
    Inputs:
        sX          name for a column for X,
        lY          list of columns for Y from df,
        df          dataframe
        sXlabel     label of X-axis
        sYlabel     label of Y-axis
        sName       name of a graph
        vMean       vector of the mean of the data

    Return value:
        plot, which is saved into a computer file
       
    """
    
    for i in range(len(lY)): 
        plt.plot(df[lY[i]],'r+')
    
    plt.plot(vMean)
    
    plt.xlabel(sXlabel)
    plt.ylabel(sYlabel)
    plt.savefig( sName + '.png', format = 'png')
    plt.show()
    
    return

def fnMean(aA, aB, dC):
    
    mean = scs.gamma(aA + aB) * scs.gamma(aA + 1/dC) / scs.gamma(aA) / scs.gamma(aA + aB + 1/dC)
    
    return mean

def Characteristics(aVector):
    
    """
    Purpose:
      Compute certain characteristic of data in a vector 

    Inputs:
      aVector    an array of data
          
    Initialize:
    
      iMean     mean
      iMed      median
      iMin      minimum
      iMax      maximum
      iKurt     kurtosis  
      iSkew     skewness    
      iStd      standard deviation
         
    Return value:
      aResults  an array with calculated characteristics
    """

    iMin  =     aVector.min().values[0]
    iMax  =     aVector.max().values[0]
    iMean =     np.mean(aVector).values[0]
    iMed  =     np.median(aVector)
    iKurt =     st.kurtosis(aVector)[0]  
    iSkew =     st.skew(aVector)[0]    
    iStd  =     aVector.std().values[0] 
      
    aResults = np.array([iMin,iMax, iMean,iMed,iKurt,iSkew,iStd])
      
    return aResults

def ReplaceValues(df, lY, iQ1, iQ2):
    
    """
    Purpose:
    Replace all values in dataframe below i1 by i1, and all values above i2 by i2.

    Inputs:
     lY          list of names of relevant columns from df
     df          dataframe
     i1          first value (lower boarder)
     i2          second value (upper boarder)
        
             
    Return value:
     dfClean     "windsorized" data frame
    """
    
    dfClean = df[lY]   
    dfClean = dfClean.clip(i1, i2)
   
    return (dfClean)

def CleanVector(df, lY):
    
    """
    Purpose:
    Create a vector of values for Characteristics function.

    Inputs:
     lY          list of names of relevant columns from df
     df          dataframe
          
    Initialize:
     vCon        vector of all the relevant colums without NaN's 
             
    Return value:
     vConNp       cleaned vector of values for Characteristics function 
    """
    
    # merging columns into one vector of dataframe
    vCon = df[lY[0]]

    for i in range(1,len(lY)):
        vCon = pd.concat([vCon,df[lY[i]]])
    
    # dropping NaNs
    vCon = vCon.dropna()
    
    # creating a NP array  
    vConNP = vCon.to_numpy()
#    vConNP = vConNP[vConNP != 0]
  
    return (vConNP)

class cMaximizeLikelihood:
    def __init__(self):
        self.x0 = []
        self.x = []
        self.a =[]
        self.b = []
        self.tx0 = []
        self.tx = []
        self.likelihoodvalue = []
        self.tcovariancematrix = []
        self.covariancematrix = []
        self.filter = []
        self.success = False

def fnGASfilter(aData, aParams):
    
    # parameters, which will be finally estimated by ML
    dOmega = aParams[0] 
    dAlpha = aParams[1]
    dBeta = aParams[2]
    dS = aParams[3]
    dC = aParams[4]
    
    # data, which is used for the observation driven model
    dfY = aData.values
    
    # initialize size
    iT = aData.shape[0]   

    # Initialize vector of loglikelihood contributions
    aLogllcontrib = np.zeros(iT)
    dM = np.nanmean(dfY.reshape((-1,1))) # mean of whole sample
    dF = np.log(dM / (1 - dM)) # starting value for time varying parameter
    aF = np.zeros(iT) # time varying parameter (in this case: mean)
    
    dScore = 0
    
    # filter variances and compute likelihood contributions by
    # a loop over the monthly observations    
    
    aA = np.zeros(iT)
    aB = np.zeros(iT)
    
    for t in range(iT):
        aMonth = aData.iloc[t].dropna().values # actual month
        aF[t] = dF
        dMu = np.exp(dF) / (1 + np.exp(dF))
        aA[t] = dS * dMu
        aB[t] = dS * (1 - dMu)
        
        if (aMonth.any() == True):
            
            # compute the likelihood contribution
            aLogllcontrib[t] = np.sum(np.log(dC) + scs.gammaln(aA[t]+aB[t]) -\
                         scs.gammaln(aA[t]) - scs.gammaln(aB[t]) +\
                         (dC*aA[t]-1)*np.log(aMonth) +\
                         (aB[t]-1)*np.log(1-np.power(aMonth, dC)))
    
            dScore = np.mean((dS*dMu*(1-dMu)) * (-scs.polygamma(0, aA[t]) + scs.polygamma(0, aB[t]) + dC * np.log(aMonth) - np.log(1-np.power(aMonth,dC))))
        
        # update the filter
        dF = dOmega + dBeta * dF + dAlpha * dScore
    
    return (aLogllcontrib, aF, aA, aB)

def fnMaximizeLikelihood(aData):

    cReturnValue = cMaximizeLikelihood()

    def LOCAL_fParameterTransform(aTheta, bShapeAsVector=False):

        r = (
            (1/(1+np.exp(-aTheta[0]))),
            (1/(1+np.exp(-aTheta[1]))),
            (1/(1+np.exp(-aTheta[2]))),
            (5/(1+np.exp(-aTheta[3]))),
            (5/(1+np.exp(-aTheta[4])))
        )
        if (bShapeAsVector == True):
            return np.append([], r)
        else:
            return r
    
    def LOCAL_fObjective(vTheta, bForAllT=False):
        # initialize the parameter values
        dOmega, dAlpha, dBeta, dS, dC = LOCAL_fParameterTransform(vTheta)

        # run the filter
        (aLogllcontrib, aF, aA, aB) = fnGASfilter(aData, [dOmega, dAlpha, dBeta, dS, dC])
        
        cReturnValue.filter = aF
        cReturnValue.a = aA
        cReturnValue.b = aB
        
        dObjValue = -np.inf
        
        if (bForAllT == True):
            dObjValue = aLogllcontrib
        else:
            dObjValue = -np.mean(aLogllcontrib)
        
        return dObjValue
    
    def LOCAL_fComputeCovarianceMatrix(vTheta):
        # compute the inverse hessian of the average log likelihood
        mH= gd.hessian_2sided(LOCAL_fObjective, vTheta)
        mCov = np.linalg.inv(mH)
        mCov = (mCov + mCov.T)/2       #  Force to be symmetric
        # compute the outer product of gradients of the average log likelihood
        mG = gd.jacobian_2sided(LOCAL_fObjective, vTheta, True)
        mG = np.dot(mG.T, mG) / aData.shape[0]
        mG = np.dot(mG, mCov)
        mCov = np.dot(mCov, mG) / aData.shape[0]
        return mCov

    # initialize starting values and return value
    dfY = aData.values
    dM = np.nanmean(dfY.reshape((-1,1))) # mean of whole sample
    dF1 = np.log(dM / (1 - dM)) # starting value for time varying parameter
    vTheta = ([dF1, 0.99, 0.001, 2.5 ,2.5])
    cReturnValue.x0 = vTheta
    cReturnValue.tx0 = LOCAL_fParameterTransform(vTheta)
    
    # do the optimization
    tSol = minimize(LOCAL_fObjective, vTheta, method='BFGS', options={'disp': True, 'maxiter':250})
    cReturnValue.success = tSol['success']
    # check for success and store results
    if (tSol['success'] != True):
        print("*** no true convergence: ",tSol['message'])
    else:
        cReturnValue.x = tSol['x']
        cReturnValue.tx = LOCAL_fParameterTransform(cReturnValue.x)
        cReturnValue.likelihoodvalue = -aData.shape[0] * tSol['fun']
        cReturnValue.covariancematrix = LOCAL_fComputeCovarianceMatrix(cReturnValue.x)
        mJ = gd.jacobian_2sided(LOCAL_fParameterTransform, cReturnValue.x, True)
        cReturnValue.tcovariancematrix = np.dot(mJ, np.dot(cReturnValue.covariancematrix, mJ.T))
    
    return cReturnValue