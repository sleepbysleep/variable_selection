import sys
import numpy as np

from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression

def sequential_feature_selection(
  x, y, pls_components=40, max_variables=502,
  validation_x=None, validation_y=None, cv=10, verbose=False
):
  if verbose:
    print('Finding the optimal variables.')

  assert(pls_components < x.shape[1])

  variable_priority = np.zeros((x.shape[1],), dtype=np.int32) # zero -> no priority
  correspond_mse = []
  
  for i in range(0, max_variables):
    variable_index = np.argwhere(variable_priority != 0)[:,0]  
    next_variable_index = np.argwhere(variable_priority == 0)[:,0]
    # print(next_variable_index)
    new_variable_info = dict()
    for j in next_variable_index:
      new_variable_index = np.append(variable_index, j)
      next_x = x[:, new_variable_index]

      model = PLSRegression(n_components=i+1)
      if cv is not None:
        y_pred = model_selection.cross_val_predict(model, next_x, y, cv=cv)
        error = mean_squared_error(y, y_pred)
      else:
        model.fit(next_x, y)
        next_valid_x = validation_x[:, new_index]
        y_pred = model.predict(next_valid_x)
        error = mean_squared_error(validation_y, y_pred)
      
      if new_variable_info == dict() or error < new_variable_info['min MSE']:
        new_variable_info['min MSE'] = error
        new_variable_info['index of min MSE'] = j
    
    # print(new_variable_info)
    # print(next_variable_index[new_variable_info['index of min MSE']])
    correspond_mse.append(new_variable_info['min MSE'])
    variable_priority[new_variable_info['index of min MSE']] = i + 1

    comp = 100 * (i + 1) / (max_variables + 1)
    if verbose:
      print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')

  # variable_index = np.argwhere(variable_priority != 0)[:,0]
  variable_index = [ np.argwhere(variable_priority == i)[0,0] for i in range(1, max_variables+1) ]

  if verbose:
    print('')
    print(' variable index: ', variable_index)
    print(' Corresponding MSE: ', correspond_mse)

  return variable_index, correspond_mse