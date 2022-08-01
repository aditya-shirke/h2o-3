from builtins import range
import sys, os
sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import numpy as np
import pandas as pd
import time
from sklearn_gbmi import *
from sklearn.ensemble import GradientBoostingClassifier


def test():
    # Setup the simulation
    n=1000  # Number of records in the simulated data
    
    # Simulate three predictors
    np.random.seed(0)
    X1 = np.random.normal(0,1,size=n)
    np.random.seed(1)
    X2 = np.random.normal(0,1,size=n)
    np.random.seed(2)
    X3 = np.random.normal(0,1,size=n)
    
    # Simulate a ranuni for assigning Y
    np.random.seed(3)
    R = np.random.uniform(0,1,size=n)
    
    # Put values in dataframe
    df = pd.DataFrame( {'id': range(1,n+1), 'X1':X1, 'X2':X2, 'X3':X3, 'R':R} )
    
    # Define the linear predictor
    B0  = -3
    B1  =  0.5
    B2  =  0.5
    B3  = -0.5
    B12 =  0
    B13 =  0
    B23 =  2
    df['LP'] = B0 + B1*df['X1'] + B2*df['X2'] + B3*df['X3'] + B12*df['X1']*df['X2'] + B13*df['X1']*df['X3'] + B23*df['X2']*df['X3']
    
    # Convert it to a probability
    df['P'] = 1/(1+np.exp(-df['LP']))
    
    # Convert it to a binary target
    df['Y'] = (df['R']<df['P']).astype(int)
    
    # Define target and predictors
    vars = ['X1', 'X2', 'X3']
    target = 'Y'
    
    # Create the data frame
    train_frame = h2o.H2OFrame(df)
    train_frame[target] = train_frame[target].asfactor()
    
    # Estimate the model
    model = H2OGradientBoostingEstimator(ntrees=5, learn_rate=0.1, max_depth=2, min_rows=1)
    model.train(x=vars, y=target, training_frame=train_frame)
    
    X = df[["X1", "X2", "X3"]]
    y = df[["Y"]]
    
    gbm_sklearn = GradientBoostingClassifier(n_estimators=5, random_state = 1234, max_depth=2, learning_rate=0.1, min_samples_leaf=1)
    
    gbm_sklearn.fit(X, np.ravel(y))

    predicted = model.predict(train_frame)["predict"].as_data_frame()
    predicted["sklearn"] = gbm_sklearn.predict(X)
    predicted['compare'] = predicted["predict"] == predicted["sklearn"]
    predicted['compare'].astype(int)
    
    print(predicted)
    print(sum(predicted["compare"]))
    # print(predicted[predicted["compare"] == False])
    
    print("REFERENCE sklearn")
    print(h_all_pairs(gbm_sklearn, X))

    print("ACTUAL H2O with frame with ignored columns")
    # Calculate the H statistic
    start_time = time.time()
    print('H between X1 and X2: {}'.format(model.h(train_frame, ['X1', 'X2'])))
    print('H between X1 and X3: {}'.format(model.h(train_frame, ['X1', 'X3'])))
    print('H between X2 and X3: {}'.format(model.h(train_frame, ['X2', 'X3'])))
    print('elapsed time:', time.time()-start_time)

    train_frame_withou_ignore_columns = h2o.H2OFrame(df[["X1", "X2", "X3"]])

    print("ACTUAL H2O with frame WITHOUT ignored columns")
    # Calculate the H statistic
    start_time = time.time()
    print('H between X1 and X2: {}'.format(model.h(train_frame_withou_ignore_columns, ['X1', 'X2'])))
    print('H between X1 and X3: {}'.format(model.h(train_frame_withou_ignore_columns, ['X1', 'X3'])))
    print('H between X2 and X3: {}'.format(model.h(train_frame_withou_ignore_columns, ['X2', 'X3'])))
    print('elapsed time:', time.time()-start_time)


if __name__ == "__main__":
  pyunit_utils.standalone_test(test)
else:
  test()
