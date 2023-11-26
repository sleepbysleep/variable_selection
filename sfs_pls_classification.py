import sys
import numpy as np

#from sklearn import datasets
from sklearn import model_selection
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression

def probility_to_binary(onehot_like_y):
  res =  np.zeros_like(onehot_like_y, dtype=np.int32)
  res[np.argmax(onehot_like_y, axis=0)] = 1
  #print(onehot_like_y, '->', res)  
  return res
  
def misclassification_rate(y_measured, y_predicted):
  total_number = y_measured.shape[0]
  missed = 0
  for y1,y2 in zip(y_measured, y_predicted):
    #print(y1, '->', y2)
    if not (y1 == y2).all():
      missed += 1
  #print('total_number:', total_number, 'missed:', missed)
  return missed / total_number

def sequential_feature_selection(
  x, y, pls_components=40, max_variables=502,
  validation_x=None, validation_y=None, cv=10, verbose=False
):
  if verbose:
    print('Finding the optimal variables.')

  assert(pls_components < x.shape[1])

  variable_priority = np.zeros((x.shape[1],), dtype=np.int32) # zero -> no priority
  correspond_mcr = []
  
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
        #print('y_pred:', y_pred)
        y_pred = np.array([probility_to_binary(x) for x in y_pred], np.int32)
        #y_pred = np.vectorize(probility_to_binary)(y_pred)
        error = misclassification_rate(y, y_pred)
      else:
        model.fit(next_x, y)
        next_valid_x = validation_x[:, new_index]
        y_pred = model.predict(next_valid_x)
        y_pred = np.array([probility_to_binary(x) for x in y_pred], np.int32)
        error = misclassification_rate(y, y_pred)
      
      if new_variable_info == dict() or error < new_variable_info['min MCR']:
        new_variable_info['min MCR'] = error
        new_variable_info['index of min MCR'] = j
    
    # print(new_variable_info)
    # print(next_variable_index[new_variable_info['index of min MCR']])
    correspond_mcr.append(new_variable_info['min MCR'])
    variable_priority[new_variable_info['index of min MCR']] = i + 1

    comp = 100 * (i + 1) / (max_variables + 1)
    if verbose:
      print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')

  # variable_index = np.argwhere(variable_priority != 0)[:,0]
  variable_index = [ np.argwhere(variable_priority == i)[0,0] for i in range(1, max_variables+1) ]

  if verbose:
    print('')
    print(' variable index: ', variable_index)
    print(' Corresponding MCR: ', correspond_mcr)

  return variable_index, correspond_mcr
