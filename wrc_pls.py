import numpy as np

from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression

def weighted_regression_coefficient(pls_model):
  # print('pls_model.coef_:', pls_model.coef_.shape)
  # y = x * pls_model.coef_ + pls_model.intercept_
  #TODO: coef_ in (n_features, n_targets) will be (n_targets, n_features)
  ascending_variable_index_array = np.argsort(np.abs(pls_model.coef_[:,0]))

  # sorted_x = x[:, ascending_variable_index_array]
  # for i in range(1, min(mse.shape[0], max_removal_variables + 1)):
  #   pls2 = PLSRegression(n_components=pls_components)
  #   if cv is not None:
  #     y_pred = model_selection.cross_val_predict(pls2, sorted_x[:,i:], y, cv=cv)
  #     mse[i] = mean_squared_error(y, y_pred)
  #   else:
  #     pls2.fit(sorted_x[:,i:], y)
  #     sorted_valid_x = validation_x[:, ascending_variable_index_array]
  #     y_pred = pls2.predict(sorted_valid_x[:,i:])
  #     mse[i] = mean_squared_error(validation_y, y_pred)
  #
  #   if mse[i] < saved_mse['min MSE']:
  #     saved_mse['min MSE'] = mse[i]
  #     saved_mse['index of min MSE'] = i
  #
  #   comp = 100 * (i+1) / min(mse.shape[0], max_removal_variables + 1)
  #   print(f'\r Grid-search the optimal variable - {comp:3.0f}% complete', end='')
  # print('')
  return np.abs(pls_model.coef_[:,0])