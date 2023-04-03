"""
Copyright © 2023 Chun Hei Michael Chan, MIPLab EPFL
"""

import random
from scipy import stats
from src.utils import *

def predict_result(classifier, trainX, trainY, valX, valY, alpha, metric='corr', ret_pred=False):
    """
    Information:
    ------------
    Do a train, prediction and evaluation of inputted set

    Parameters
    ----------
    classifier::[sklearn.predictor]
        Model used for regression
    trainX    ::[2darray<float>]
        gradients train set
    trainY    ::[2darray<float>]
        label (i.e personality index) train set
    valX      ::[2darray<float>]
        gradients validation set
    valY      ::[2darray<float>]
        label (i.e personality index) validation set
    alpha     ::[float]
        weighting for lasso
    metric    ::[String]
    ret_pred  ::[Bool]

    Returns
    -------
    res       ::[float]
        scoring of classifier on validation
    coef      ::[1darray<float>]
    prediction::[1darray<float>]
        on validation set
    """

    clf        = classifier(alpha=alpha)
    clf.fit(trainX, trainY)
    prediction = clf.predict(valX)

    
    if metric == 'corr':
        res = stats.pearsonr(prediction, valY).statistic
    else:
        res = clf.score(valX,valY)

    if ret_pred:
        return res, clf.coef_ , prediction
    else:
        return res, clf.coef


def combination_predict(classifier, hparam_range, X, y, kfold, metric='corr', rd_state=10):
    """
    Information:
    ------------
    Find best parameter a specific pair of
    the issue is we have a overfitting problem, the solutions are not necessarily found, 
    and so we consider a solution to have a postiive R2 values on validaiton

    Parameters
    ----------
    classifier  ::[sklearn.predictor]
        Model used for regression
    hparam_range::[1darray<float>]
        Alpha values for weighting in L1 loss
    X           ::[2darray<float>]
    y           ::[1darray<float>]
    kfold       ::[int]
        test fold size
    metric      ::[String]
    rd_state    ::[int]

    Returns
    -------
    fold_results::[list<res,coef,pred,valY,corr>]
        all folds' set of results
    """

    np.random.seed(rd_state)
    # kfold needs to divisor of 30
    fold_results = []

    # k-fold splits
    splits    = np.random.choice(np.arange(len(y)),replace=False, size=(kfold,30//kfold))

    for j in range(kfold):
        # reset for this fold the best correlation search within the hyper parameter ranges
        best_res  = -1
        best_coef = None
        best_pred = None
        best_corr = -1

        val_idx   = splits[j]
        train_idx = np.array(list(set(np.arange(len(X))) - set(val_idx)))

        trainX, valX    = X[train_idx], X[val_idx] 
        trainY, valY    = y[train_idx], y[val_idx]

        for a in hparam_range:
            res, coef, prediction =  predict_result(classifier, trainX, trainY, valX, valY, a, metric=metric, ret_pred=True)

            if res > best_res:
                best_res  = res
                best_coef = coef
                best_pred = prediction
                best_corr = stats.pearsonr(prediction, valY).statistic
                
        fold_results.append([best_res, best_coef, best_pred, valY, best_corr])
    
    return fold_results

def sample_solution(classifier, hparam_range, X, y, kfold, nrepeat, null=False, metric='corr', rd_state=10):
    """
    Information:
    ------------
    Example of classifier results on a given set with k-folding results

    Parameters
    ----------
    classifier  ::[sklearn.predictor]
        Model used for regression
    hparam_range::[1darray<float>]
        Alpha values for weighting in L1 loss
    X           ::[2darray<float>]
    y           ::[1darray<float>]
    kfold       ::[int]
        test fold size
    nrepeat     ::[int]
    metric      ::[String]
    rd_state    ::[int]

    Returns
    -------
    fold_results::[list<res,coef,corr>]
        all folds' set of results
    """

    # kfold needs to divisor of 30
    np.random.seed(rd_state)
    fold_results = []
    for k in range(nrepeat):

        # if null distribution then shuffle y
        if null:
            sy = deepcopy(y)
            np.random.shuffle(sy)
        else:
            sy = deepcopy(y)
        # k-fold splits included
        fold_p = combination_predict(classifier, hparam_range, 
                                            X, sy, kfold, metric=metric, rd_state=k)
        best_res  = np.asarray(fold_p)[:,0]
        best_coef = np.asarray(fold_p)[:,1]
        best_corr = np.asarray(fold_p)[:,-1]    
        for b,_ in enumerate(best_res):
            if best_res[b] > 0:
                fold_results.append([best_res[b], best_coef[b], best_corr[b]])

    return fold_results