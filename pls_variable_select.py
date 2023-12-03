import sys
import numpy as np
import multiprocessing
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import qr

def mean_categorical_cross_entropy(y_true, y_pred):
  '''
  Normalize y_pred which don't have minus element and their sum have 1.0
  ---
  y_true : Y true set in 2D of # validation samples x # classes
  y_pred : Y predict set in 2D of # validation samples x # classes
  ---
  return Mean Categorical Cross Entropy
  '''
  y_norm = y_pred.copy()
  for i, yh in enumerate(y_norm):
    for j in range(yh.shape[0]):
      if yh[j] < 0:
        y_norm[i, :] += (-yh[j] + 0.01) / (yh.shape[0] - 1)
        y_norm[i, j] = 0.01

  cce = np.array([-np.sum(y * np.log(yh)) for y, yh in zip(y_true, y_norm)])
  return np.mean(cce)

def apply_pls_regression(components, xcalib, ycalib, cv, xvalid, yvalid):
  '''
  Perform PLS calibration and validation in the way of K-fold Cross Validation or extra validation set.
  ---
  components : the number of PLS components
  xcalib : X calibration set in 2D of # calibration samples x # variables
  ycalib : Y calibration set in 1D of # calibration samples
  cv : k-fold cross validation
  xvalid : X validation set in 2D of # validation samples x # variables
  yvalid : Y validation set in 1D of # validation samples
  ---
  return : Mean Squared Error according to K-fold Cross Validation or Extra validation
  '''
  pls = PLSRegression(n_components=components)
  if cv is not None:
    ypred = model_selection.cross_val_predict(pls, xcalib, ycalib, cv=cv)
    mse = mean_squared_error(ycalib, ypred)
  else:
    pls.fit(xcalib, ycalib)
    ypred = pls.predict(xvalid)
    mse = mean_squared_error(yvalid, ypred)
  return mse

def apply_pls_da(components, xcalib, ycalib, cv, xvalid, yvalid):
  '''
  Perform PLS-DA calibration and validation in the way of K-fold Cross Validation or extra validation set.
  ---
  components : the number of PLS-DA components
  xcalib : X calibration set in 2D of # calibration samples x # variables
  ycalib : Y calibration set in 2D of # calibration samples x # classes
  cv : k-fold cross validation
  xvalid : X validation set in 2D of # validation samples x # variables
  yvalid : Y validation set in 2D of # validation samples x # classes
  ---
  return : Mean Categorical Cross Entropy
  '''
  pls = PLSRegression(n_components=components)
  if cv is not None:
    ypred = model_selection.cross_val_predict(pls, xcalib, ycalib, cv=cv)
    mcce = mean_categorical_cross_entropy(ycalib, ypred)
  else:
    pls.fit(xcalib, ycalib)
    ypred = pls.predict(xvalid)
    mcce = mean_categorical_cross_entropy(yvalid, ypred)

  return mcce

class PLSVariableSelector:
  @classmethod
  def __variable_score_x_std(cls, pls, variable_std):
    # print('pls1.coef_:', pls1.coef_.shape) # y = x * pls.coef_ + pls.intercept_
    # TODO: coef_ in (n_features, n_targets) will be (n_targets, n_features)
    # FIXME: determine the score which is from PLS coefficient or PLS coefficient + standard deviation
    # return np.abs(pls.coef_[0, :] * variable_std) # for regression
    score = np.abs(pls.coef_ * variable_std) # for classification
    return np.mean(score, axis=0)

  @classmethod
  def getRemovalIndicesByFixedScoreXStd(
          cls, xcalib, ycalib, components, max_removal_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the removeable variables accumulated of a candidate by sequential searching
     based on fixed variable score (PLS coefficients + standard deviation of variable)
    ---
    xcalib : X calibration set in 2D(# of calib. samples x # of variables)
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    components : the number of PLS components
    max_removal_variables : the maximum limitation for searching removeable variables
    xvalid : X validation set in 2D(# of valid. samples x # of variables)
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of removeable variables, list of Losses, the index of min loss
    '''
    assert (components <= xcalib.shape[1])

    removeable_variables = xcalib.shape[1] - components
    assert (removeable_variables > 0)

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    loss_log = np.zeros(
      shape=(min(max_removal_variables, removeable_variables) + 1,), dtype=np.float64
    )

    if verbose:
      print('Finding the index of removeable variables.')
      print(' the number of max removealble variables:', loss_log.shape[0]-1)

    pls1 = PLSRegression(n_components=components)
    if cv is not None:
      y_pred = model_selection.cross_val_predict(pls1, xcalib, ycalib, cv=cv)
      loss_log[0] = loss_func(ycalib, y_pred)
      pls1.fit(xcalib, ycalib)
    else:
      pls1.fit(xcalib, ycalib)
      y_pred = pls1.predict(xvalid)
      loss_log[0] = loss_func(yvalid, y_pred)

    # TODO: coef_ in (n_features, n_targets) will be (n_targets, n_features)
    # FIXME: determine the score which is from PLS coefficient or PLS coefficient + standard deviation
    # variable_score = np.mean(np.abs(pls.coef_), axis=0)
    variable_std = np.std(xcalib, axis=0)
    variable_score = cls.__variable_score_x_std(pls1, variable_std)
    ascending_variable_score_indices = np.argsort(variable_score, axis=0)

    #######################################################
    # sorted_x = x[:, ascending_variable_score_indices]
    # for i in range(1, min(loss_log.shape[0], max_removal_variables + 1)):
    #   pls2 = PLSRegression(n_components=pls_components)
    #   if cv is not None:
    #     y_pred = model_selection.cross_val_predict(pls2, sorted_x[:,i:], y, cv=cv)
    #     loss_log[i] = loss_func(y, y_pred)
    #   else:
    #     pls2.fit(sorted_x[:,i:], y)
    #     sorted_valid_x = validation_x[:, ascending_variable_score_indices]
    #     y_pred = pls2.predict(sorted_valid_x[:,i:])
    #     loss_log[i] = loss_func(validation_y, y_pred)
    #
    #   if loss_log[i] < saved_loss_log['min LOSS']:
    #     saved_mse['min LOSS'] = loss_log[i]
    #     saved_mse['index of min LOSS'] = i
    #
    #   comp = 100 * (i+1) / min(loss_log.shape[0], max_removal_variables + 1)
    #   print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')
    # print('')
    #######################################################
    args = []
    for i in range(1, loss_log.shape[0]):
      args.append([
        components,
        xcalib[:, ascending_variable_score_indices[i:]],
        ycalib.copy(),
        cv,
        None if xvalid is None else xvalid[:, ascending_variable_score_indices[i:]],
        None if yvalid is None else yvalid.copy()
      ])

    with multiprocessing.Pool() as p:
      loss_log[1:] = p.starmap(apply_pls, args)
    #######################################################

    if verbose:
      print(' # of variables to be discarded: ', np.argmin(loss_log))
      print(' Corresponding LOSS: ', np.min(loss_log))

    return ascending_variable_score_indices, loss_log, np.argmin(loss_log)

  @classmethod
  def getRemovalIndicesByUpdatingScoreXStd(
          cls, xcalib, ycalib, components, max_removal_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the variables accumulated of a candidate by sequential searching
     based on updating variable score (PLS coefficients + variable standard deviation)
    ---
    xcalib : X calibration set in 2D(# of calib. samples x # of variables)
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    components : the number of PLS components
    max_removal_variables : the maximum limitation for searching removeable variables
    xvalid : X validation set in 2D(# of valid. samples x # of variables)
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of removeable variables, list of losses, the index of min loss
    '''
    assert (components <= xcalib.shape[1])

    removeable_variables = xcalib.shape[1] - components
    assert (removeable_variables > 0)

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    loss_log = np.zeros(
      shape=(min(max_removal_variables, removeable_variables) + 1,), dtype=np.float64
    )

    if verbose:
      print('Finding the index of removeable variables.')
      print(' the number of max removealble variables:', loss_log.shape[0]-1)

    pls = PLSRegression(n_components=components)
    if cv is not None:
      y_pred = model_selection.cross_val_predict(pls, xcalib, ycalib, cv=cv)
      loss_log[0] = loss_func(ycalib, y_pred)
      pls.fit(xcalib, ycalib)
    else:
      pls.fit(xcalib, ycalib)
      y_pred = pls.predict(xvalid)
      loss_log[0] = loss_func(yvalid, y_pred)

    saved_loss = {'min LOSS': loss_log[0], 'index of min LOSS': 0}
    variable_std = np.std(xcalib, axis=0)
    variable_priority = np.zeros(shape=(xcalib.shape[1],), dtype=np.int32)
    next_variable_indices = np.argwhere(variable_priority == 0)[:, 0]

    for i in range(1, loss_log.shape[0]):
      # TODO: coef_ in (n_features, n_targets) will be (n_targets, n_features)
      # FIXME: determine the score which is from PLS coefficient or PLS coefficient + standard deviation
      # variable_score = np.mean(np.abs(pls.coef_), axis=0)
      variable_score = cls.__variable_score_x_std(pls, variable_std[next_variable_indices])
      removal_index = np.argmin(variable_score)

      variable_priority[next_variable_indices[removal_index]] = i
      # next_variable_indices = np.argwhere(variable_priority == 0)[:, 0]
      next_variable_indices = np.delete(next_variable_indices, removal_index, axis=0)

      next_x = xcalib[:, next_variable_indices]
      # pls = PLSRegression(n_components=components)
      if cv is not None:
        y_pred = model_selection.cross_val_predict(pls, next_x, ycalib, cv=cv)
        loss_log[i] = loss_func(ycalib, y_pred)
        pls.fit(next_x, ycalib)
      else:
        pls.fit(next_x, ycalib)
        y_pred = pls.predict(xvalid[:, next_variable_indices])
        loss_log[i] = loss_func(yvalid, y_pred)

      if loss_log[i] < saved_loss['min LOSS']:
        saved_loss['min LOSS'] = loss_log[i]
        saved_loss['index of min LOSS'] = i

      if verbose:
        comp = 100 * (i + 1) / loss_log.shape[0]
        print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')

    if verbose:
      print('')
      print(' # of variables to be discarded: ', saved_loss['index of min LOSS'])
      print(' Corresponding LOSS: ', saved_loss['min LOSS'])

    removeable_variable_indices = np.array([i for i in np.argsort(variable_priority) if variable_priority[i] != 0])
    return removeable_variable_indices, loss_log, saved_loss['index of min LOSS']

  @classmethod
  def __variable_importance_in_projection(cls, pls_model):
    t = pls_model.x_scores_
    # w = pls_model.x_weights_
    w = pls_model.x_rotations_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
      weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
      # print('weight:', weight)
      # print('s:', s.T)
      # print('->', s.T @ weight)
      vips[i] = np.sqrt(p * (s[:,0].T @ weight) / total_s)
      # vips[i] = np.sqrt(p * s[:,0].dot(weight) / total_s)
    return vips

  @classmethod
  def getRemovalIndicesByFixedVariableImportanceInProjection(
          cls, xcalib, ycalib, components, max_removal_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the removeable variables accumulated of a candidate by sequential searching
     based on fixed Variable Importance in Projection(VIP)
    ---
    xcalib : X calibration set in 2D(# of calib. samples x # of variables)
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    components : the number of PLS components
    max_removal_variables : the maximum limitation for searching removeable variables
    xvalid : X validation set in 2D(# of valid. samples x # of variables)
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of removeable variables, list of losses, the index of min loss
    '''
    assert (components <= xcalib.shape[1])

    removeable_variables = xcalib.shape[1] - components
    assert (removeable_variables > 0)

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    loss_log = np.zeros(
      shape=(min(max_removal_variables, removeable_variables) + 1,), dtype=np.float64
    )

    if verbose:
      print('Finding the index of removeable variables.')
      print(' the number of max removealble variables:', loss_log.shape[0] - 1)

    pls1 = PLSRegression(n_components=components)
    if cv is not None:
      y_pred = model_selection.cross_val_predict(pls1, xcalib, ycalib, cv=cv)
      loss_log[0] = loss_func(ycalib, y_pred)
      pls1.fit(xcalib, ycalib)
    else:
      pls1.fit(xcalib, ycalib)
      y_pred = pls1.predict(xvalid)
      loss_log[0] = loss_func(yvalid, y_pred)

    variable_score = cls.__variable_importance_in_projection(pls1)
    ascending_variable_score_indices = np.argsort(variable_score, axis=0)

    #######################################################
    # sorted_x = x[:, ascending_variable_score_indices]
    # for i in range(1, min(loss_log.shape[0], max_removal_variables + 1)):
    #   pls2 = PLSRegression(n_components=pls_components)
    #   if cv is not None:
    #     y_pred = model_selection.cross_val_predict(pls2, sorted_x[:,i:], y, cv=cv)
    #     loss_log[i] = loss_func(y, y_pred)
    #   else:
    #     pls2.fit(sorted_x[:,i:], y)
    #     sorted_valid_x = validation_x[:, ascending_variable_score_indices]
    #     y_pred = pls2.predict(sorted_valid_x[:,i:])
    #     loss_log[i] = loss_func(validation_y, y_pred)
    #
    #   if loss_log[i] < saved_mse['min MSE']:
    #     saved_mse['min MSE'] = loss_log[i]
    #     saved_mse['index of min MSE'] = i
    #
    #   comp = 100 * (i+1) / min(loss_log.shape[0], max_removal_variables + 1)
    #   print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')
    # print('')
    #######################################################
    args = []
    for i in range(1, loss_log.shape[0]):
      args.append((
        components,
        xcalib[:, ascending_variable_score_indices[i:]],
        ycalib.copy(),
        cv,
        None if xvalid is None else xvalid[:, ascending_variable_score_indices[i:]],
        None if yvalid is None else yvalid.copy()
      ))

    with multiprocessing.Pool() as p:
      loss_log[1:] = p.starmap(apply_pls, args)
    #######################################################

    if verbose:
      print(' # of variables to be discarded: ', np.argmin(loss_log))
      print(' Corresponding LOSS: ', np.min(loss_log))

    return ascending_variable_score_indices, loss_log, np.argmin(loss_log)

  @classmethod
  def getRemovalIndicesByUpdatingVariableImportanceInProjection(
          cls, xcalib, ycalib, components, max_removal_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the variables accumulated of a candidate by sequential searching
     based on updating variable score (PLS coefficients + variable standard deviation)
    ---
    xcalib : X calibration set in 2D(# of calib. samples x # of variables)
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    components : the number of PLS components
    max_removal_variables : the maximum limitation for searching removeable variables
    xvalid : X validation set in 2D(# of valid. samples x # of variables)
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of removeable variables, list of losses, the index of min loss
    '''
    assert (components <= xcalib.shape[1])

    removeable_variables = xcalib.shape[1] - components
    assert (removeable_variables > 0)

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    loss_log = np.zeros(
      shape=(min(max_removal_variables, removeable_variables) + 1,), dtype=np.float64
    )

    if verbose:
      print('Finding the index of removeable variables.')
      print(' the number of max removealble variables:', loss_log.shape[0]-1)

    pls = PLSRegression(n_components=components)
    if cv is not None:
      y_pred = model_selection.cross_val_predict(pls, xcalib, ycalib, cv=cv)
      loss_log[0] = loss_func(ycalib, y_pred)
      pls.fit(xcalib, ycalib)
    else:
      pls.fit(xcalib, ycalib)
      y_pred = pls.predict(xvalid)
      loss_log[0] = loss_func(yvalid, y_pred)

    saved_loss = {'min LOSS': loss_log[0], 'index of min LOSS': 0}
    variable_std = np.std(xcalib, axis=0)
    variable_priority = np.zeros(shape=(xcalib.shape[1],), dtype=np.int32)
    next_variable_indices = np.argwhere(variable_priority == 0)[:, 0]

    for i in range(1, loss_log.shape[0]):
      variable_score = cls.__variable_importance_in_projection(pls)
      removal_index = np.argmin(variable_score)

      variable_priority[next_variable_indices[removal_index]] = i
      # next_variable_indices = np.argwhere(variable_priority == 0)[:, 0]
      next_variable_indices = np.delete(next_variable_indices, removal_index, axis=0)

      next_x = xcalib[:, next_variable_indices]
      # pls = PLSRegression(n_components=components)
      if cv is not None:
        y_pred = model_selection.cross_val_predict(pls, next_x, ycalib, cv=cv)
        loss_log[i] = loss_func(ycalib, y_pred)
        pls.fit(next_x, ycalib)
      else:
        pls.fit(next_x, ycalib)
        y_pred = pls.predict(xvalid[:, next_variable_indices])
        loss_log[i] = loss_func(yvalid, y_pred)

      if loss_log[i] < saved_loss['min LOSS']:
        saved_loss['min LOSS'] = loss_log[i]
        saved_loss['index of min LOSS'] = i

      if verbose:
        comp = 100 * (i + 1) / loss_log.shape[0]
        print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')

    if verbose:
      print('')
      print(' # of variables to be discarded: ', saved_loss['index of min LOSS'])
      print(' Corresponding LOSS: ', saved_loss['min LOSS'])

    removeable_variable_indices = np.array([i for i in np.argsort(variable_priority) if variable_priority[i] != 0])
    return removeable_variable_indices, loss_log, saved_loss['index of min LOSS']

  @staticmethod
  def getRemovalIndicesBySequentialSearch(
          xcalib, ycalib, components, max_removal_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the variables accumulated of a candidate by sequential searching for the min MSE(Mean Square Error)
    ---
    xcalib : X calibration set in 2D(# of calib. samples x # of variables)
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    components : the number of PLS components
    max_removal_variables : the maximum limitation for searching removeable variables
    xvalid : X validation set in 2D(# of valid. samples x # of variables)
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of removeable variables, log of losses, the index of min loss
    '''
    assert(components <= xcalib.shape[1])

    removeable_variables = xcalib.shape[1] - components
    assert(removeable_variables > 0)

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    loss_log = np.zeros(
      shape=(min(max_removal_variables, removeable_variables) + 1,), dtype=np.float64
    )

    if verbose:
      print('Finding the index of removeable variables.')
      print(' the number of max removealble variables:', loss_log.shape[0]-1)

    loss_log[0] = apply_pls(components, xcalib, ycalib, cv, xvalid, yvalid)
    saved_loss = { 'min LOSS': loss_log[0], 'index of min LOSS': 0 }

    variable_priority = np.zeros(shape=(xcalib.shape[1],), dtype=np.int32)
    next_variable_indices = np.argwhere(variable_priority == 0)[:,0]

    for i in range(1, loss_log.shape[0]):
      ###########################################################################
      # for j in range(next_variable_indices.shape[0]):
      #   # new_index = np.ma.array(next_variable_indices, mask=False)
      #   # new_index.mask[j] = True
      #   # next_x = x[:,new_index.compressed()]
      #   new_index = np.delete(next_variable_indices, j, axis=0)
      #   next_x = x[:, new_index]
      #
      #   pls2 = PLSRegression(n_components=pls_components)
      #   if cv is not None:
      #     y_pred = model_selection.cross_val_predict(pls2, next_x, y, cv=cv)
      #     error = mean_squared_error(y, y_pred)
      #   else:
      #     pls2.fit(next_x, y)
      #     next_valid_x = validation_x[:,new_index]
      #     y_pred = pls2.predict(next_valid_x)
      #     error = mean_squared_error(validation_y, y_pred)
      #   if error < next_variable_info['min LOSS']:
      #     next_variable_info['min LOSS'] = error
      #     next_variable_info['index of min LOSS'] = j
      ###########################################################################
      args = []
      for j in range(next_variable_indices.shape[0]):
        new_index = np.delete(next_variable_indices, j, axis=0)
        args.append((
          components, xcalib[:, new_index], ycalib.copy(), cv,
          None if xvalid is None else xvalid[:, new_index],
          None if yvalid is None else yvalid.copy()
        ))

      next_variable_info = { 'min LOSS': np.float64('inf'), 'index of min LOSS': None }
      with multiprocessing.Pool() as p:
        res = p.starmap(apply_pls, args)
        for j, error in enumerate(res):
          if error < next_variable_info['min LOSS']:
            next_variable_info['min LOSS'] = error
            next_variable_info['index of min LOSS'] = j
      ###########################################################################

      loss_log[i] = next_variable_info['min LOSS']
      if loss_log[i] < saved_loss['min LOSS']:
        saved_loss['min LOSS'] = loss_log[i]
        saved_loss['index of min LOSS'] = i

      variable_priority[next_variable_indices[next_variable_info['index of min LOSS']]] = i
      next_variable_indices = np.delete(next_variable_indices, next_variable_info['index of min LOSS'], axis=0)
      # new_index = np.ma.array(next_variable_indices, mask=False)
      # new_index.mask[next_variable_info[1]] = True
      # next_variable_indices = new_index.compressed()

      if verbose:
        comp = 100 * (i + 1) / loss_log.shape[0]
        print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')

    if verbose:
      print('')
      print(' # of variables to be discarded: ', saved_loss['index of min LOSS'])
      print(' Corresponding LOSS: ', saved_loss['min LOSS'])

    removal_variable_indices = np.array([ i for i in np.argsort(variable_priority) if variable_priority[i] != 0 ])
    return removal_variable_indices, loss_log, saved_loss['index of min LOSS']

  @classmethod
  def __chain_variables_by_qr_projection(cls, X, variable_index, max_variables):
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


  @classmethod
  def getIndicesBySuccessiveProjectionAlgorithm(
          cls, xcalib, ycalib, min_variables, max_variables,
          xvalid=None, yvalid=None, cv=10, loss='mse', verbose=False
  ):
    '''
    Find out the variables accumulated of a candidate by Successive Projection Algorithm(SPA)
    ---
    xcalib : X calibration set in 2D of # calib. samples x # variables
    ycalib : Y calibration set
      In the case of regression, 1D(# of calib. samples) 
      In the case of classification, 2D(# of calib. samples x # of classes)
    min_variables : the minimum limit for searching optimal variables
    max_variables : the maximum limit for searching optimal variables
    xvalid : X validation set in 2D of # valid. samples x # variables
    yvalid : Y validation set
      In the case of regression, 1D(# of valid. samples)
      In the case of classification, 2D(# of valid. samples x # of classes)
    cv : k-fold cross validation
    loss : Error Measurement Method
      In the case of regression, 'Mean Squared Error', or 'MSE', 
      In the case of regression, 'Mean Categorical Cross Entropy', or 'MCCE'
    ---
    return : the index of feasible variables, Mean Square Errors, the index of min MSE
    '''
    total_samples, total_variables = xcalib.shape
    if max_variables is None:
      if xvalid is None:
        max_variables = min(total_samples - 1, total_variables)
      else:
        max_variables = min(total_samples - 2, total_variables)
    assert (max_variables <= min(total_samples - 1, total_variables))

    if loss.lower() == 'mean squared error' or loss.lower() == 'mse':
      loss_func = mean_squared_error
      apply_pls = apply_pls_regression
    elif loss.lower() == 'mean categorical cross entropy' or loss.lower() == 'mcce':
      loss_func = mean_categorical_cross_entropy
      apply_pls = apply_pls_da
    else:
      raise ValueError("Unknown Loss Function")

    if verbose:
      print('Finding the index of feasible variables.')
      print(' the number of max feasible variables:', max_variables)

    # preparation the index list of available variables
    variable_indices_set = np.zeros(shape=(max_variables, total_variables), dtype=np.int32)
    # args = []
    # for j in range(total_variables):
    #   args.append((xcalib.copy(), j, max_variables))
    #
    # with multiprocessing.Pool() as p:
    #   res = p.starmap(cls.__chain_variables_by_qr_projection, args)
    #   print(res.shape)

    for j in range(total_variables):
      variable_indices_set[:, j] = cls.__chain_variables_by_qr_projection(
        xcalib, j, max_variables
      )
    if verbose: print('variable_indices_set:\n', variable_indices_set)

    mse = np.ones((max_variables + 1, total_variables)) * float('inf')
    for i in range(min_variables, max_variables+1):
      args = []
      for j in range(total_variables):
        variable_indices = variable_indices_set[:i, j]
        args.append((
          variable_indices.shape[0],
          xcalib[:, variable_indices],
          ycalib.copy(),
          cv,
          None if xvalid is None else xvalid[:, variable_indices],
          None if yvalid is None else yvalid.copy()
        ))
      with multiprocessing.Pool() as p:
        mse[i,:] = p.starmap(apply_pls, args)

    min_mse_of_variable_sets = np.min(mse, axis=0)
    optimal_variable_count_of_variable_set = np.argmin(mse, axis=0)
    optimal_variable_set_index = np.argmin(optimal_variable_count_of_variable_set)
    if verbose:
      print('MSEs:\n', mse)
      print('min MSEs of variable sets:\n', min_mse_of_variable_sets)
      print('optimal variable count of variable_set:\n', optimal_variable_count_of_variable_set)
      print('optimal variable set index:\n', optimal_variable_set_index)

    optimal_variable_indices = variable_indices_set[
      :optimal_variable_count_of_variable_set[optimal_variable_set_index],
      optimal_variable_set_index
    ]
    # selective_variable_index = possible_variable_index[:, min_variable_index]
    if verbose: print('optimal variable indices:\n', optimal_variable_indices)

    ## sort selective_variable_index according to their weight
    pls = PLSRegression(n_components=optimal_variable_indices.shape[0])
    pls.fit(xcalib[:,optimal_variable_indices], ycalib)
    variable_std = np.std(xcalib[:,optimal_variable_indices], axis=0)
    variable_score = cls.__variable_score_x_std(pls, variable_std)
    ascending_variable_score_indices = np.argsort(variable_score, axis=0)
    descending_variable_score_indices = ascending_variable_score_indices[::-1]
    if verbose:
      print('Ascending Variable Score Indices:', ascending_variable_score_indices)
      print('Descending Variable Score Indices:', descending_variable_score_indices)

    ## accumulate RMS errors while adding relative important variable
    loss_log = np.empty(len(optimal_variable_indices))
    for i in range(len(optimal_variable_indices)):
      variable_indices = optimal_variable_indices[descending_variable_score_indices[:i+1]]
      loss_log[i] = apply_pls(
        i+1, xcalib[:, variable_indices], ycalib, cv,
        None if xvalid is None else xvalid[:, variable_indices],
        None if yvalid is None else yvalid.copy()
      )

    return optimal_variable_indices[descending_variable_score_indices], loss_log, np.argmin(loss_log, axis=0)
