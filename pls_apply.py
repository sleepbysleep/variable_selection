import numpy as np
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

def mean_categorical_cross_entropy(y_true, y_pred):
  '''
  Normalize y_pred which will not have negative element and their sum will be 1.0
  ---
  y_true : Y true set in 2D of # validation samples x # classes
  y_pred : Y predict set in 2D of # validation samples x # classes
  ---
  return Mean Categorical Cross Entropy
  '''
  y_norm = y_pred.copy()
  for i, yh in enumerate(y_norm):
    for j in range(yh.shape[0]):
      if yh[j] <= 0:
        y_norm[i, :] += -yh[j] + 0.01
        # y_norm[i, j] = 0.01
  assert((y_norm > 0.0).all())

  y_norm = y_norm / np.sum(y_norm, axis=1).reshape(-1, 1)
  # print(np.sum(y_norm, axis=1))
  # print(np.sum(y_norm, axis=1)[2])
  # assert((np.sum(y_norm, axis=1) == 1.0).all())
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