# variable_selection
Variable selection for NIR spectral analysis(regression and classification) based on WRC, VIP, SFS, and SPA

# WRC(Weighted Regression Coefficient)
TODO: add description

# VIP(Variable Importance in Projection)
TODO: add description

# SFS(Sequential Feature Selection)
TODO: add description

# SPA(Successive Projections Algorithm)
TODO: add description

# Usage
## Regression
```bash
$ python3 variable_selection_regression.py
samle.feature_names
 ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
sample.data
 [[ 0.03807591  0.05068012  0.06169621 ... -0.00259226  0.01990749
  -0.01764613]
 [-0.00188202 -0.04464164 -0.05147406 ... -0.03949338 -0.06833155
  -0.09220405]
 [ 0.08529891  0.05068012  0.04445121 ... -0.00259226  0.00286131
  -0.02593034]
 ...
 [ 0.04170844  0.05068012 -0.01590626 ... -0.01107952 -0.04688253
   0.01549073]
 [-0.04547248 -0.04464164  0.03906215 ...  0.02655962  0.04452873
  -0.02593034]
 [-0.04547248 -0.04464164 -0.0730303  ... -0.03949338 -0.00422151
   0.00306441]]
sample.target
 [151.  75. 141. 206. 135.  97. 138.  63. 110. 310. 101.  69. 179. 185.
 ...
  49.  64.  48. 178. 104. 132. 220.  57.]
/.../_pls.py:503: FutureWarning: The attribute `coef_` will be transposed in version 1.3 to be consistent with other linear models in scikit-learn. Currently, `coef_` has a shape of (n_features, n_targets) and in the future it will have a shape of (n_targets, n_features).
  warnings.warn(
### Weighted Regression Coefficient(WRC):
 [0.00618993 0.14829793 0.3214639  0.20059396 0.48986798 0.29480733
 0.06248344 0.1094929  0.46457492 0.0418192 ]
 corresponding priority variable index:
 [4 8 2 5 3 1 7 6 9 0]
### Variable Importance in Projection(VIP):
 [0.48981785 0.63296189 1.53475149 1.08532397 0.73738729 0.79707229
 0.93573914 1.02206738 1.38169913 0.90424625]
 corresponding priority variable index:
 [2 8 3 7 6 9 5 4 1 0]
### SFS -> priority index, and corresponding MSE:
 priority index:
 [2, 8, 3, 6, 1, 4, 5, 7, 9, 0]
 corresponding MSE:
 [0.658775033228952, 0.5453132711423678, 0.5253106822016119, 0.5150814939752454, 0.500373593411865, 0.4982043233651625, 0.4980014803525826, 0.49942372228544823, 0.5010897289463909, 0.5057503741879567]
### SPA:
possible_variable_index:
 [[0 1 2 3 4 5 6 7 8 9]
 [6 4 1 6 1 1 4 0 1 1]
 [4 2 0 5 2 0 0 1 0 5]
 [1 0 5 0 0 2 1 3 5 0]
 [3 9 9 1 9 9 3 2 2 2]
 [9 3 6 9 3 6 9 9 3 6]
 [2 6 3 2 6 3 2 4 6 3]
 [8 8 8 8 8 8 8 8 9 8]
 [7 7 7 7 7 7 7 6 7 7]]
square_errors:
 [[         inf          inf          inf          inf          inf
           inf          inf          inf          inf          inf]
 [         inf          inf          inf          inf          inf
           inf          inf          inf          inf          inf]
 [366.55084274 427.12189512 293.72874924 314.26863603 427.12189512
  434.19965187 353.75446136 360.21951351 303.73101107 381.58878805]
 [350.66826543 292.97411403 291.92236749 315.08685207 292.97411403
  425.47244443 350.66826543 355.48568899 304.20876757 381.21963551]
 [342.67118186 292.08510283 293.31954351 315.88361316 292.08510283
  293.31954351 342.67118186 302.68863077 305.53763508 380.38828395]
 [292.4606163  283.1846395  283.28560504 299.90595107 283.1846395
  283.28560504 292.4606163  252.29166854 242.82962857 283.28560504]
 [285.78870325 267.24448573 265.80040666 289.8591604  267.24448573
  265.80040666 285.78870325 251.40527683 232.65968783 265.80040666]
 [246.77062209 246.77062209 247.3432364  247.3432364  246.77062209
  247.3432364  246.77062209 247.73653299 223.54766546 247.3432364 ]
 [223.3437257  223.3437257  223.89606039 223.89606039 223.3437257
  223.89606039 223.3437257  223.53620226 223.89606039 223.89606039]
 [223.95268692 223.95268692 224.81604726 224.81604726 223.95268692
  224.81604726 223.95268692 223.95268692 224.81604726 224.81604726]]
min_square_errors:
 [223.3437257  223.3437257  223.89606039 223.89606039 223.3437257
 223.89606039 223.3437257  223.53620226 223.54766546 223.89606039]
selective_variable_count:
 [8 8 8 8 8 8 8 8 7 8]
min_variable_index:
 8
selective_variable_index:
 [8 1 0 5 2 3 6]
/.../_pls.py:503: FutureWarning: The attribute `coef_` will be transposed in version 1.3 to be consistent with other linear models in scikit-learn. Currently, `coef_` has a shape of (n_features, n_targets) and in the future it will have a shape of (n_targets, n_features).
 Feasible Features:
 ['bmi' 's5']
 Feasible Variable Index:
 [2 8]
 Available Features:
 ['bmi' 's5' 'bp' 's3' 'sex' 's2' 'age']
 Available Variable Index:
 [2 8 3 6 1 5 0]
```

## Classification
```bash
$ python3 variable_selection_classification.py
sample.feature_names
 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
sample.data
 [[5.1 3.5 1.4 0.2]
...
 [5.9 3.  5.1 1.8]]
sample.target
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 ...
 2 2]
one-hot style
 [[1 0 0]
 ...
 [0 0 1]]
X:
 [[-9.00681170e-01  1.01900435e+00 -1.34022653e+00 -1.31544430e+00]
 ...
 [ 6.86617933e-02 -1.31979479e-01  7.62758269e-01  7.90670654e-01]]
Y:
 [[1 0 0]
 ...
 [0 0 1]]
/.../_pls.py:503: FutureWarning: The attribute `coef_` will be transposed in version 1.3 to be consistent with other linear models in scikit-learn. Currently, `coef_` has a shape of (n_features, n_targets) and in the future it will have a shape of (n_targets, n_features).
  warnings.warn(
### Weighted Regression Coefficient(WRC):
 [0.05467702 0.1058492  0.39658681 0.04380788]
 corresponding priority variable index:
 [2 1 0 3]
### Variable Importance in Projection(VIP):
 [0.91579838 0.91215631 1.06271796 1.09540619]
 corresponding priority variable index:
 [3 2 0 1]
### SFS -> priority index, and corresponding MCR:
 priority index:
 [3, 1, 2, 0]
 corresponding MCR:
 [0.34, 0.22666666666666666, 0.24, 0.23333333333333334]
### SPA:
possible_variable_index:
 [[0 1 2 3]
 [1 0 1 1]
 [3 3 0 0]
 [2 2 3 2]]
x_entropy_errors:
 [[       inf        inf        inf        inf]
 [       inf        inf        inf        inf]
 [0.50942344 0.50942344 0.37680952 0.36558822]
 [0.36616363 0.36616363 0.37499475 0.36616363]
 [0.34451786 0.34451786 0.34451786 0.34451786]]
min_errors:
 [0.34451786 0.34451786 0.34451786 0.34451786]
selective_variable_count:
 [4 4 4 4]
min_variable_index:
 0
selective_variable_index:
 [0 1 3 2]
/.../_pls.py:503: FutureWarning: The attribute `coef_` will be transposed in version 1.3 to be consistent with other linear models in scikit-learn. Currently, `coef_` has a shape of (n_features, n_targets) and in the future it will have a shape of (n_targets, n_features).
 Feasible Features:
 ['petal length (cm)' 'sepal width (cm)']
 Feasible Variable Index:
 [2 1]
 Available Features:
 ['petal length (cm)' 'sepal width (cm)' 'sepal length (cm)'
 'petal width (cm)']
 Available Variable Index:
 [2 1 0 3]

```
