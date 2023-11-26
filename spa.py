# Modified from http://gitee.com/aBugsLife/SPA
import pandas as pd
import numpy as np
from scipy.linalg import qr, inv, pinv
import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import log_loss
#from progress.bar import Bar
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut

class SPA:
  def __prepare_parameters(self, xcalib, xvalid, max_variables):
    total_calibs,total_variables = xcalib.shape
    if max_variables is None:
      if xvalid is None:
        max_variables = min(total_calibs - 1, total_variables)
      else:
        max_variables = min(total_calibs - 2, total_variables)
    assert(max_variables <= min(total_calibs - 1, total_variables))
    return total_calibs,total_variables,max_variables

  def __normalize(self, xcalib, autoscaling=True):
    # x data normalization including mean-centering
    # and unit standard devation(optional)
    normalization_factor = None
    if autoscaling:
      normalization_factor = np.std(xcalib, ddof=1, axis=0).reshape(1, -1)[0]
    else:
      normalization_factor = np.ones((1, xcalib.shape[1]))[0]

    xnorm = np.empty_like(xcalib)
    for j in range(xcalib.shape[1]):
      x = xcalib[:, j]
      xnorm[:, j] = (x - np.mean(x)) / normalization_factor[j]
    return xnorm

  def __chain_variables_by_qr_projection(self, X, variable_index, max_variables):
    '''
    X with variables in the way of column vectors
    For interesting variable indcied by 'variable_index',
    this performs variable chaining to have largest projection score.
    '''
    xscaled = X.copy()
    norms = np.sum(X ** 2, axis=0)
    max_norm = np.amax(norms)
    xscaled[:, variable_index] = X[:, variable_index] * 2 * max_norm / norms[variable_index]

    Q,R,reference_variable_index = qr(xscaled, overwrite_a=True, pivoting=True)
    # print('Q with column vector of eigen vector:\n', Q)
    # print('R upper diagonal matrix:\n', R)
    # print('reference_variable_index:\n', reference_variable_index)
    return reference_variable_index[:max_variables].T

  def __mlr_validate(
      self, xcalib, ycalib, available_variable_index,
      xvalid=None, yvalid=None
  ):
    total_valids = 0 if xvalid is None else xvalid.shape[0]
    
    ypred = error = None
    if total_valids > 0:
      ones_with_xcalib = np.hstack([
        np.ones((xcalib.shape[0], 1)),
        xcalib[:, available_variable_index]
      ])
      # b = np.linalg.lstsq(ones_with_xcalib, ycalib.flatten(), rcond=None)[0]
      b = np.linalg.lstsq(ones_with_xcalib, ycalib, rcond=None)[0]
      ones_with_xvalid = np.hstack([
        np.ones((xvalid.shape[0], 1)),
        xvalid[:, available_variable_index]
      ])
      ypred = ones_with_xvalid.dot(b)
      # error = yvalid.flatten() - ypred
      error = yvalid - ypred
    else:
      ypred = np.zeros((xcalib.shape[0], ))
      for i in range(xcalib.shape[0]):
        x = np.delete(xcalib, i, axis=0)[:, available_variable_index]
        y = np.delete(ycalib, i, axis=0)
        ones_with_x = np.hstack([np.ones((x.shape[0], 1)), x])
        b = np.linalg.lstsq(ones_with_x, y, rcond=None)[0]
        ypred[i] = np.hstack([
          np.ones(1), xcalib[i, available_variable_index]
        ]).dot(b)
      error = ycalib - ypred

    return ypred, error

  def __pls_validate(
      self, xcalib, ycalib, available_variable_index,
      xvalid=None, yvalid=None
  ):
    total_valids = 0 if xvalid is None else xvalid.shape[0]

    ypred = error = None
    if total_valids > 0:
      pls = PLSRegression(n_components=available_variable_index.shape[0])
      pls.fit(xcalib[:, available_variable_index], ycalib)
      ypred = pls.predict(xvalid[:, available_variable_index])
      error = yvalid - ypred
    else:
      ypred = np.zeros((xcalib.shape[0], ))
      for i in range(xcalib.shape[0]):
        x = np.delete(xcalib, i, axis=0)[:, available_variable_index]
        y = np.delete(ycalib, i, axis=0)
        pls = PLSRegression(n_components=available_variable_index.shape[0])
        pls.fit(x, y)
        ypred[i] = pls.predict(xcalib[i, available_variable_index].reshape(1,-1))[0]
      error = ycalib - ypred

    return ypred, error

  def __plsda_validate(
      self, xcalib, ycalib, available_variable_index,
      xvalid=None, yvalid=None
  ):
    total_valids = 0 if xvalid is None else xvalid.shape[0]
    
    ypred = error = None
    if total_valids > 0:
      try:
        plsda = PLSRegression(n_components=available_variable_index.shape[0])
        plsda.fit(xcalib[:, available_variable_index], ycalib)
      except:
        print("Invalid the number of components with ",var_sel)
        plsda = PLSRegression(n_components=1)
        plsda.fit(xcalib[:, available_variable_index], ycalib)
      ypred = plsda.predict(xvalid[:, available_variable_index])
      # log_loss (a.k.a. categorical cross entropy for loss function of multi-classifiction)
      error = np.array([-np.sum(y * np.log(yh)) for y,yh in zip(yvalid, ypred)])
    else:
      ypred = []
      loo = LeaveOneOut()
      for train, test in loo.split(xcalib):
        plsda = PLSRegression(n_components=available_variable_index.shape[0])
        # print(test)
        xtrain = xcalib[train, :]
        plsda.fit(xtrain[:, available_variable_index], ycalib[train])
        xtest = xcalib[test]
        ypred.append(plsda.predict(xtest[:, available_variable_index])[0])

      ### FIXME : negative probability -> scaling -> non-negative probability
      for i,yh in enumerate(ypred):
        for j in range(yh.shape[0]):
          if yh[j] < 0:
            ypred[i] += (-yh[j] + 0.01) / (yh.shape[0]-1) 
            ypred[i][j] = 0.01
          # print(yhat[i])
      ypred = np.array(ypred)
      error = np.array([-np.sum(y * np.log(yh)) for y, yh in zip(ycalib, ypred)])

    return ypred, error

  def spa_mlr(
      self, xcalib, ycalib, min_variables=1, max_variables=None,
      xvalid=None, yvalid=None, autoscaling=True, withplot=True
  ):
    '''
    successive projection algorithm based on multivariate linear regression
    xcalib : X calibration in calibration samples x variables
    ycalib : y calibration in calibration samples x 1
    min_variables : the minimum number of selective variables >= 2
    max_variables : the maximum number of selective variables <= variables
    xvalid : X validation in validation samples x variables
    yvalid : y validation in validation samples x 1
    autoscaling : on-off switch of mean-centering and unit std normalization
    withplot : on-off switch for drawing graph after f-test 
    '''
    # Parameter preparation
    total_calibs,total_variables,max_variables = self.__prepare_parameters(
      xcalib, xvalid, max_variables
    )
    
    # x data normalization including mean-centering
    # and unit standard devation(optional)
    xnorm = self.__normalize(xcalib, autoscaling)

    # preparation the index list of available variables
    possible_variable_index = np.zeros(
      (max_variables, total_variables), dtype=np.int32
    )
    for j in range(total_variables):
      possible_variable_index[:, j] = self.__chain_variables_by_qr_projection(
        xnorm, j, max_variables
      )
    print('possible_variable_index:\n', possible_variable_index)

    square_errors = np.ones((max_variables + 1, total_variables)) * float('inf') 
    for j in range(total_variables):
      for i in range(min_variables, max_variables+1):
        _,e = self.__mlr_validate(
          xcalib, ycalib,
          possible_variable_index[:i, j],
          xvalid, yvalid
        )
        square_errors[i,j] = e.dot(e)

    min_square_errors = np.min(square_errors, axis=0)
    selective_variable_count = np.argmin(square_errors, axis=0)
    min_variable_index = np.argmin(selective_variable_count)
    print('square_errors:\n', square_errors)
    print('min_square_errors:\n', min_square_errors)
    print('selective_variable_count:\n', selective_variable_count)
    print('min_variable_index:\n', min_variable_index)

    selective_variable_index = possible_variable_index[
      :selective_variable_count[min_variable_index],
      min_variable_index
    ]
    # selective_variable_index = possible_variable_index[:, min_variable_index]
    print('selective_variable_index:\n', selective_variable_index)

    ## sort selective_variable_index according to their weight
    ones_with_xcalib = np.hstack([
      np.ones((xcalib.shape[0],1)),
      xcalib[:, selective_variable_index]
    ])
    b = np.linalg.lstsq(ones_with_xcalib, ycalib, rcond=None)[0]
    standard_deviation = np.std(ones_with_xcalib, ddof=1, axis=0)
    relevance = np.abs(b * standard_deviation.T)[1:]
    ascending_relevance_index = np.argsort(relevance, axis=0)
    descending_relevance_index = ascending_relevance_index[::-1].reshape(1,-1)[0]

    ## accumulate RMS errors while adding relative important variable 
    e = None
    square_errors = np.empty(len(selective_variable_index))
    for i in range(len(selective_variable_index)):
      variable_index = selective_variable_index[descending_relevance_index[:i+1]]
      _,e = self.__mlr_validate(xcalib, ycalib, variable_index, xvalid, yvalid)
      square_errors[i] = np.conj(e).T.dot(e)
    rms_errors = np.sqrt(square_errors / len(e))

    ## f-test
    alpha = 0.25
    dof = len(e)
    fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
    min_square_error = np.min(square_errors)
    square_error_limit = min_square_error * fcrit

    # 找到不明显比 PRESS_scree_min 大的最小变量
    i_crit = np.min(np.nonzero(square_errors < square_error_limit))
    i_crit = max(min_variables, i_crit)

    available_variable_index = selective_variable_index[
      descending_relevance_index[:i_crit]
    ]

    if withplot:
      # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
      # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
      fig1 = plt.figure()
      plt.xlabel('Number of variables included in the model')
      plt.ylabel('RMSE')
      plt.title(
        'Final number of selected variables:{}(RMSE={})'.format(
          len(available_variable_index), rms_errors[i_crit]
        )
      )
      plt.plot(rms_errors)
      plt.scatter(i_crit, rms_errors[i_crit], marker='s', color='r')
      plt.grid(True)

      fig2 = plt.figure()
      plt.plot(xcalib[0, :])
      plt.scatter(
        available_variable_index,
        xcalib[0, available_variable_index],
        marker='s', color='r'
      )
      plt.legend(['First calibration object', 'Selected variables'])
      plt.xlabel('Variable index')
      plt.grid(True)
      plt.show()

    return available_variable_index, selective_variable_index[descending_relevance_index]
    #return available_variable_index, possible_variable_index[:, min_variable_index]

  def spa_pls(
      self, xcalib, ycalib, min_variables=1, max_variables=None,
      xvalid=None, yvalid=None, autoscaling=True, withplot=True
  ):
    '''
    successive projection algorithm based on multivariate linear regression
    xcalib : X calibration in calibration samples x variables
    ycalib : y calibration in calibration samples x 1
    min_variables : the minimum number of selective variables >= 2
    max_variables : the maximum number of selective variables <= variables
    xvalid : X validation in validation samples x variables
    yvalid : y validation in validation samples x 1
    autoscaling : on-off switch of mean-centering and unit std normalization
    withplot : on-off switch for drawing graph after f-test 
    '''
    # parameter preparation
    total_calibs,total_variables,max_variables = self.__prepare_parameters(
      xcalib, xvalid, max_variables
    )

    # x data normalization including mean-centering
    # and unit standard devation(optional)
    xnorm = self.__normalize(xcalib, autoscaling)

    # preparation the index list of available variables
    possible_variable_index = np.zeros(
      (max_variables, total_variables), dtype=np.int32
    )
    for j in range(total_variables):
      possible_variable_index[:, j] = self.__chain_variables_by_qr_projection(
        xnorm, j, max_variables
      )
    print('possible_variable_index:\n', possible_variable_index)

    square_errors = np.ones((max_variables + 1, total_variables)) * float('inf') 
    for j in range(total_variables):
      for i in range(min_variables, max_variables+1):
        _,e = self.__pls_validate(
          xcalib, ycalib,
          possible_variable_index[:i, j],
          xvalid, yvalid
        )
        square_errors[i,j] = e.dot(e)

    min_square_errors = np.min(square_errors, axis=0)
    selective_variable_count = np.argmin(square_errors, axis=0)
    min_variable_index = np.argmin(selective_variable_count)
    print('square_errors:\n', square_errors)
    print('min_square_errors:\n', min_square_errors)
    print('selective_variable_count:\n', selective_variable_count)
    print('min_variable_index:\n', min_variable_index)

    selective_variable_index = possible_variable_index[
      :selective_variable_count[min_variable_index],
      min_variable_index
    ]
    # selective_variable_index = possible_variable_index[:, min_variable_index]
    print('selective_variable_index:\n', selective_variable_index)

    ## sort selective_variable_index according to their weight
    pls = PLSRegression(n_components=selective_variable_index.shape[0])
    pls.fit(xcalib[:,selective_variable_index], ycalib)
    standard_deviation = np.std(
      xcalib[:,selective_variable_index], ddof=1, axis=0
    )
    relevance = np.abs(pls.coef_[:,0] * standard_deviation.T)
    ascending_relevance_index = np.argsort(relevance, axis=0)
    descending_relevance_index = ascending_relevance_index[::-1].reshape(1,-1)[0]

    ## accumulate RMS errors while adding relative important variable 
    e = None
    square_errors = np.empty(len(selective_variable_index))
    for i in range(len(selective_variable_index)):
      variable_index = selective_variable_index[descending_relevance_index[:i+1]]
      _,e = self.__mlr_validate(xcalib, ycalib, variable_index, xvalid, yvalid)
      square_errors[i] = np.conj(e).T.dot(e)
    rms_errors = np.sqrt(square_errors / len(e))

    ## f-test
    alpha = 0.25
    dof = len(e)
    fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
    min_square_error = np.min(square_errors)
    square_error_limit = min_square_error * fcrit

    # 找到不明显比 PRESS_scree_min 大的最小变量
    i_crit = np.min(np.nonzero(square_errors < square_error_limit))
    i_crit = max(min_variables, i_crit)

    available_variable_index = selective_variable_index[
      descending_relevance_index[:i_crit]
    ]

    if withplot:
      # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
      # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
      fig1 = plt.figure()
      plt.xlabel('Number of variables included in the model')
      plt.ylabel('RMSE')
      plt.title(
        'Final number of selected variables:{}(RMSE={})'.format(
          len(available_variable_index), rms_errors[i_crit]
        )
      )
      plt.plot(rms_errors)
      plt.scatter(i_crit, rms_errors[i_crit], marker='s', color='r')
      plt.grid(True)

      fig2 = plt.figure()
      plt.plot(xcalib[0, :])
      plt.scatter(
        available_variable_index,
        xcalib[0, available_variable_index],
        marker='s', color='r'
      )
      plt.legend(['First calibration object', 'Selected variables'])
      plt.xlabel('Variable index')
      plt.grid(True)
      plt.show()

    return available_variable_index, selective_variable_index[descending_relevance_index]
    #return available_variable_index, possible_variable_index[:, min_variable_index]

  def spa_plsda(
      self, xcalib, ycalib, min_variables=1, max_variables=None,
      xvalid=None, yvalid=None, autoscaling=True, withplot=True
  ):
    '''
    successive projection algorithm based on multivariate linear regression
    xcalib : X calibration in calibration samples x variables
    ycalib : y calibration in calibration samples x classes
    min_variables : the minimum number of selective variables >= 2
    max_variables : the maximum number of selective variables <= variables
    xvalid : X validation in validation samples x variables
    yvalid : y validation in validation samples x classes
    autoscaling : on-off switch of mean-centering and unit std normalization
    withplot : on-off switch for drawing graph after f-test 
    '''
    # parameter preparation
    total_calibs,total_variables,max_variables = self.__prepare_parameters(
      xcalib, xvalid, max_variables
    )
    # x data normalization including mean-centering
    # and unit standard devation(optional)
    xnorm = self.__normalize(xcalib, autoscaling)
    
    # preparation the index list of available variables
    possible_variable_index = np.zeros(
      (max_variables, total_variables), dtype=np.int32
    )
    for j in range(total_variables):
      possible_variable_index[:, j] = self.__chain_variables_by_qr_projection(
        xnorm, j, max_variables
      )
    print('possible_variable_index:\n', possible_variable_index)

    x_entropy_errors = float('inf') * np.ones((max_variables+1, total_variables))
    for j in range(total_variables):
      for i in range(min_variables, max_variables+1):
        _,e = self.__plsda_validate(
          xcalib, ycalib, possible_variable_index[:i, j], xvalid, yvalid
        )
        # categorical cross entropy for loss function of one-hot style output
        x_entropy_errors[i,j] = np.mean(e) 
    
    min_errors = np.min(x_entropy_errors, axis=0)
    selective_variable_count = np.argmin(x_entropy_errors, axis=0)
    min_variable_index = np.argmin(selective_variable_count)
    print('x_entropy_errors:\n', x_entropy_errors)
    print('min_errors:\n', min_errors)
    print('selective_variable_count:\n', selective_variable_count)
    print('min_variable_index:\n', min_variable_index)

    selective_variable_index = possible_variable_index[
      :selective_variable_count[min_variable_index],
      min_variable_index
    ]
    # selective_variable_index = possible_variable_index[:, min_variable_index]
    print('selective_variable_index:\n', selective_variable_index)

    ## sort selective_variable_index according to their weight
    plsda = PLSRegression(n_components=selective_variable_index.shape[0])
    plsda.fit(xcalib[:,selective_variable_index], ycalib)
    standard_deviation = np.std(
      xcalib[:,selective_variable_index], ddof=1, axis=0
    )
    relevance = np.abs(plsda.coef_[:,0] * standard_deviation.T)
    ascending_relevance_index = np.argsort(relevance, axis=0)
    descending_relevance_index = ascending_relevance_index[::-1].reshape(1,-1)[0]

    ## accumulate RMS errors while adding relative important variable 
    e = None
    x_entropy_errors = np.empty(len(selective_variable_index))
    for i in range(len(selective_variable_index)):
      variable_index = selective_variable_index[descending_relevance_index[:i+1]]
      _,e = self.__plsda_validate(xcalib, ycalib, variable_index, xvalid, yvalid)
      x_entropy_errors[i] = np.sum(e)
    mean_errors = np.sqrt(x_entropy_errors / len(e))

    ## f-test
    alpha = 0.25
    dof = len(e)
    fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
    min_xentropy_error = np.min(x_entropy_errors)
    xentropy_error_limit = min_xentropy_error * fcrit

    # 找到不明显比 PRESS_scree_min 大的最小变量
    i_crit = np.min(np.nonzero(x_entropy_errors < xentropy_error_limit))
    i_crit = max(min_variables, i_crit)

    available_variable_index = selective_variable_index[
      descending_relevance_index[:i_crit]
    ]

    if withplot:
      # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
      # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
      fig1 = plt.figure()
      plt.xlabel('Number of variables included in the model')
      plt.ylabel('Mean Cross Entropy Error')
      plt.title(
        'Final number of selected variables:{}(RMSE={})'.format(
          len(available_variable_index), mean_errors[i_crit]
        )
      )
      plt.plot(mean_errors)
      plt.scatter(i_crit, mean_errors[i_crit], marker='s', color='r')
      plt.grid(True)

      fig2 = plt.figure()
      plt.plot(xcalib[0, :])
      plt.scatter(
        available_variable_index,
        xcalib[0, available_variable_index],
        marker='s', color='r'
      )
      plt.legend(['First calibration object', 'Selected variables'])
      plt.xlabel('Variable index')
      plt.grid(True)
      plt.show()

    return available_variable_index, selective_variable_index[descending_relevance_index]
    #return available_variable_index, possible_variable_index[:, min_variable_index]

  def __repr__(self):
    return "SPA()"
