import sys
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pls_variable_select import PLSVariableSelector

def main(argv):
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
  fig.canvas.manager.set_window_title(
    "Find removal variables based on PLS coeff x variable Std, VIP, and sequential searching,\
     and optimal variabled based on Successive Projection Algorithm"
  )
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

  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
