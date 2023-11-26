import sys
import numpy as np

from sklearn import datasets, model_selection
#from sklearn.metrics import f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from wrc_pls import weighted_regression_coefficient
from vip_pls import variable_importance_in_projection
from sfs_pls_classification import sequential_feature_selection
from spa import SPA

def main(argv):
  '''
  PLS-DA : PLS for classification
  https://stackoverflow.com/questions/18390150/pls-da-algorithm-in-python
  '''
  sample = datasets.load_iris()
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
  
  pls_da = PLSRegression(n_components=X.shape[1]).fit(X, Y)

  ### WRC -> priority -> corresponding index of variable
  variable_priority = weighted_regression_coefficient(pls_da)
  print(
    "### Weighted Regression Coefficient(WRC):\n",
    variable_priority
  )

  priority_index = np.argsort(variable_priority)[::-1]
  print(
    " corresponding priority variable index:\n",
    priority_index
  )

  ### VIP -> priority -> corresponding index of variable
  variable_priority = variable_importance_in_projection(pls_da)
  print(
    "### Variable Importance in Projection(VIP):\n",
    variable_priority
  )
  # array([1.43159475, 1.20362379, 1.05294437, 1.26626086, 1.33362676,
  #        0.03800398, 0.06995447, 0.06942297])

  priority_index = np.argsort(variable_priority)[::-1]
  print(
    " corresponding priority variable index:\n",
    priority_index
  )

  ### SFS
  print("### SFS -> priority index, and corresponding MCR:")
  priority_index, correspond_mcr = sequential_feature_selection(
    X, Y, 2, X.shape[1], None, None, 10, False
  )
  print(" priority index:\n", priority_index)
  print(" corresponding MCR:\n", correspond_mcr)
  
  print("### SPA:")
  feasible_variable_index, available_index = SPA().spa_plsda(
    X, Y, min_variables=2, max_variables=X.shape[1], xvalid=None, yvalid=None, autoscaling=True
  )
  # feasible_variable_index, available_index = SPA().spa(
  #   X, Y, m_min=2, m_max=X.shape[1]-1, Xval=None, yval=None, autoscaling=True
  # )
  print(
    " Feasible Features:\n",
    np.array(sample.feature_names)[feasible_variable_index]
  )
  # print(absorbances[var_sel])
  print(
    " Feasible Variable Index:\n", feasible_variable_index
  )
  print(
    " Available Features:\n",
    np.array(sample.feature_names)[available_index]
  )
  print(
    " Available Variable Index:\n", available_index
  )

  # y_predicted = pls_da.predict(X)
  # print('predicted\n', y_predicted)
  
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
