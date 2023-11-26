import numpy as np

# from sklearn import datasets
# from sklearn import model_selection
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.cross_decomposition import PLSRegression

def variable_importance_in_projection(pls_model):
  t = pls_model.x_scores_
  # w = pls_model.x_weights_
  w = pls_model.x_rotations_
  q = pls_model.y_loadings_
  p, h = w.shape
  vips = np.zeros((p,))
  s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
  total_s = np.sum(s)
  for i in range(p):
      weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
      vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
  return vips