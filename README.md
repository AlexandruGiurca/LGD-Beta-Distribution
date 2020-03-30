# Modelling Loss Given Default (LGD) with a Time-Varying Beta Distribution

## 1. Introduction

In the context of credit portfolio losses, loss given default (LGD) is the proportion of the exposure that will be lost if a default occurs. Uncertainty regarding the actual LGD is an important source of credit portfolio risk in addition to default risk. Therefore the LGD is is a key ingredient of current financial risk management and regulation. Precise evaluation of this parameter is important not only for bank to calculate their regulatory capital but also for investors to price risky bonds and credit derivatives. 

### 1.1 Problem of actual Modelling Approaches

In practice the uncertainty in the LGD rates of defaulted obligors is assumed to be a **static** beta random variable independent for each obligor. Such a modeling strategy is highly risky if the properties of LGDs actually vary over time. For example, losses could be on average higher in situations where default risk is also higher, thus exacerbating total expected losses defined as the probability of default times the LGD. 

### 1.2 Aim 

The goal is to implement a model that improves on current industry standards and is (relatively) easy to implement and estimate: **Generalized Autoregressive Score (GAS) Model** for a **time-varying beta distribution**. GAS models are a flexible class of *observation driven time-varying parameter* models characterized by a parametric conditional observation density.

## 2. Implementation

Estimation of parameters is performed by maximum likelihood (ML). For an observed time series of LGD's and by adopting the standard *Prediction Error Decomposition*, one can obtain the Maximum Likelihood Estimator. Given the parametric assumption of GAS models, predictions for several steps ahead are computed.

 ## 3. Application 
I use a sample panel data set of LGDs for corporate bond data obtained from Moody’s to model the average monthly LGD. After deriving the GAS model theoretically, I fit it to the data, estimate parameters and asymptotically robust standard errors.  
