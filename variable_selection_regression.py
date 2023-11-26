import sys
import numpy as np

from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from wrc_pls import weighted_regression_coefficient
from vip_pls import variable_importance_in_projection
from sfs_pls_regression import sequential_feature_selection
from spa import SPA

def main(argv):
  ### Begining of program
  sample = datasets.load_diabetes()
  print('samle.feature_names\n', sample.feature_names)
  print('sample.data\n', sample.data)
  print('sample.target\n', sample.target)
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

  pls = PLSRegression(n_components=X.shape[1]).fit(X, Y)

  ### WRC -> priority -> corresponding index of variable
  variable_priority = weighted_regression_coefficient(pls)
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
  variable_priority = variable_importance_in_projection(pls)
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
  print("### SFS -> priority index, and corresponding MSE:")
  priority_index, correspond_mse = sequential_feature_selection(
    X, Y, 2, X.shape[1], None, None, 10, False
  )
  print(" priority index:\n", priority_index)
  print(" corresponding MSE:\n", correspond_mse)

  print("### SPA:")
  #feasible_variable_index, available_index = SPA().spa(
  #  X, Y, m_min=2, m_max=X.shape[1]-1, Xval=None, yval=None, autoscaling=True
  #)
  
  #feasible_variable_index, available_index = SPA().spa_mlr(
  #  X, Y, min_variables=2, max_variables=X.shape[1]-1,
  #  xvalid=None, yvalid=None, autoscaling=True
  #)
  
  feasible_variable_index, available_index = SPA().spa_pls(
    X, Y, min_variables=2, max_variables=X.shape[1]-1,
    xvalid=None, yvalid=None, autoscaling=True
  )
  
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
  
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
