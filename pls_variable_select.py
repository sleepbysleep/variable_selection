import sys
import numpy as np
import multiprocessing
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import qr
from pls_apply import mean_categorical_cross_entropy, apply_pls_da, apply_pls_regression

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

def test_variable_selection_for_pls_regression():
  from sklearn import datasets
  from sklearn.preprocessing import StandardScaler
  import matplotlib
  matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from pls_variable_select import PLSVariableSelector

  sample = datasets.load_diabetes()
  print('samle.feature_names\n', sample.feature_names)
  print('sample.data\n', sample.data.shape)
  print('sample.target\n', sample.target.shape)
  X = sample.data
  Y = sample.target

  ### Preprocessing
  xscaler = StandardScaler()
  xscaler.fit(X)
  X = xscaler.transform(X)
  # preprocessX = xScaler.inverse_transform(preprocessX)

  yscaler = StandardScaler()
  Y = Y.reshape(-1, 1)
  yscaler.fit(Y)
  Y = yscaler.transform(Y)
  Y = Y.reshape(-1)

  ### Extra validation set
  x_calib, x_valid, y_calib, y_valid = train_test_split(X, Y, test_size=0.33)
  cv = None

  ### 10-fold cross validation
  # x_calib = X
  # y_calib = Y
  # x_valid = None
  # y_valid = None
  # cv = 10

  fig = plt.figure(figsize=(16,9))
  fig.canvas.manager.set_window_title("Variable Selection")
  fig.tight_layout()
  plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.5, hspace=0.3)

  print("### getRemovalIndicesByFixedScoreXStd ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getRemovalIndicesByFixedScoreXStd(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mse_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Fixed Weight (PLS Coeff x Variable Std)')
    # ax1.text(0.5, np.max(mse), , color='r')

  print("### getRemovalIndicesByUpdatingScoreXStd ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getRemovalIndicesByUpdatingScoreXStd(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mse_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Updating Weight (PLS Coeff x Variable Std)')

  print("### getRemovalIndicesByFixedVariableImportanceInProjection ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getRemovalIndicesByFixedVariableImportanceInProjection(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mse_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 0), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Fixed Variable Importance in Projection')

  print("### getRemovalIndicesByUpdatingVariableImportanceInProjection ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getRemovalIndicesByUpdatingVariableImportanceInProjection(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mse_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Updating Variable Importance in Projection')

  print("### getRemovalIndicesBySequentialSearch ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getRemovalIndicesBySequentialSearch(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mse_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Sequential Searching for min MSE')

  print("### getIndicesBySuccessiveProjectionAlgorithm ###")
  variable_indices, mse, min_mse_index = PLSVariableSelector.getIndicesBySuccessiveProjectionAlgorithm(
    x_calib, y_calib, min_variables=3, max_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MSE', verbose=False
  )

  optimal_variable_indices = variable_indices[:min_mse_index+1]
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  removeable_variable_indices = np.array(set(range(x_calib.shape[1])) - set(optimal_variable_indices))
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)
    ax1.plot(mse, '-', color='blue', mfc='blue')
    ax1.plot(min_mse_index, mse[min_mse_index], 'P', ms=10, mfc='red')
    ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of Selected Variables')
    ax1.set_ylabel('Mean Square Error')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    # ax1.set_xlim(left=-1)
    ax1.grid(True)
    ax1.set_title('Successive Projection Algorithm')

  plt.show()

def test_variable_selection_for_pls_da():
  from sklearn import datasets
  from sklearn.preprocessing import StandardScaler
  import matplotlib
  matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from pls_variable_select import PLSVariableSelector

  '''
  PLS-DA : PLS for classification
  https://stackoverflow.com/questions/18390150/pls-da-algorithm-in-python
  '''
  sample = datasets.load_wine()
  print('sample.feature_names\n', sample.feature_names)
  print('sample.data\n', sample.data)
  print('sample.target\n', sample.target)
  X = sample.data
  Y_label = sample.target # as one-hot style

  Y = np.zeros((Y_label.size, Y_label.max()+1), dtype=np.int32)
  Y[np.arange(Y_label.size), Y_label] = 1
  print('one-hot style\n', Y)  ### Preprocessing
  xscaler = StandardScaler()
  #yscaler = StandardScaler()

  xscaler.fit(X)
  X = xscaler.transform(X)
  # preprocessX = xScaler.inverse_transform(preprocessX)

  print('X:\n', X)
  print('Y:\n', Y)

  # # Extra validation set
  # x_calib, x_valid, y_calib, y_valid = train_test_split(X, Y, test_size=0.33)
  # cv = None

  # 10-fold cross validation
  x_calib = X
  y_calib = Y
  cv = 10
  x_valid = None
  y_valid = None

  fig = plt.figure(figsize=(16,9))
  fig.canvas.manager.set_window_title(
    "Find removal variables based on PLS coeff x variable Std, VIP, and sequential searching,\
     and optimal variabled based on Successive Projection Algorithm"
  )
  fig.tight_layout()
  plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.5, hspace=0.3)

  print("### getRemovalIndicesByFixedScoreXStd ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getRemovalIndicesByFixedScoreXStd(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mcce_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Categorical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Fixed Weight (PLS Coeff x Variable Std)')
    # ax1.text(0.5, np.max(mse), , color='r')

  print("### getRemovalIndicesByUpdatingScoreXStd ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getRemovalIndicesByUpdatingScoreXStd(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mcce_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 2), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Categrical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Updating Weight (PLS Coeff x Variable Std)')

  print("### getRemovalIndicesByFixedVariableImportanceInProjection ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getRemovalIndicesByFixedVariableImportanceInProjection(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mcce_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 0), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Categorical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Fixed Variable Importance in Projection')

  print("### getRemovalIndicesByUpdatingVariableImportanceInProjection ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getRemovalIndicesByUpdatingVariableImportanceInProjection(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mcce_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 2), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Categorical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Updating Variable Importance in Projection')

  print("### getRemovalIndicesBySequentialSearch ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getRemovalIndicesBySequentialSearch(
    x_calib, y_calib, components=5, max_removal_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  removeable_variable_indices = variable_indices[:min_mcce_index]
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  optimal_variable_indices = np.array(set(range(x_calib.shape[1])) - set(removeable_variable_indices))
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (0, 4), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    # ax1.set_xticks(range(0, len(mse)), [ str(x) for x in range(1, len(mse)+1)])
    ax1.set_xlabel('# of removeable variables')
    ax1.set_ylabel('Mean Categorical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax1.set_title('Sequential Searching for min MCCE')

  print("### getIndicesBySuccessiveProjectionAlgorithm ###")
  variable_indices, mcce, min_mcce_index = PLSVariableSelector.getIndicesBySuccessiveProjectionAlgorithm(
    x_calib, y_calib, min_variables=3, max_variables=x_calib.shape[1],
    xvalid=x_valid, yvalid=y_valid, cv=cv, loss='MCCE', verbose=False
  )

  optimal_variable_indices = variable_indices[:min_mcce_index + 1]
  print("Optimal Variable's Indices:\n", optimal_variable_indices)

  removeable_variable_indices = np.array(set(range(x_calib.shape[1])) - set(optimal_variable_indices))
  print("Removeable Variable's Indices:\n", removeable_variable_indices)

  with plt.style.context(('ggplot')):
    ax1 = plt.subplot2grid((2, 6), (1, 4), rowspan=1, colspan=2)
    ax1.plot(mcce, '-', color='blue', mfc='blue')
    ax1.plot(min_mcce_index, mcce[min_mcce_index], 'P', ms=10, mfc='red')
    ax1.set_xticks(range(0, len(mcce)), [str(x) for x in range(1, len(mcce) + 1)])
    ax1.set_xlabel('# of Selected Variables')
    ax1.set_ylabel('Mean Categorical Cross Entropy')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    # ax1.set_xlim(left=-1)
    ax1.grid(True)
    ax1.set_title('Successive Projection Algorithm')

  plt.show()

if __name__ == '__main__':
  test_variable_selection_for_pls_regression()
  test_variable_selection_for_pls_da()
  sys.exit(0)