from likelihood import generate_series, regression
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def create_index(data_series, start = '2018'):
    n = data_series.shape[1]
    stop = np.datetime64('2018') + np.timedelta64(n, 'M')
    date = np.arange(start+'-01', stop, dtype='datetime64[M]')
    return date
    
def convert_to_df(data_series):
    return pd.DataFrame(data_series[:, :, 0].T, index = create_index(data_series))

def create_train_test_set(data_series, p_train = 0.7):
    n_steps = data_series.shape[1]
    
    train_ = int(data_series.shape[0])
    train_init = int(data_series.shape[1]*p_train)
    x_train = data_series[:train_, :train_init-5]
    y_train = data_series[:train_, train_init-5:train_init, 0]
    
    train = [x_train, y_train]
    
    x_test = data_series[train_:, :train_init-5]
    y_test = data_series[train_:, train_init-5:train_init, 0]
    
    test = [x_test, y_test]
    
    return train, test

def forecasting(model, train, m, values):
    n = int(train[0].shape[1])
    n_periods = int((m - n)/5.)
    data_pred = np.copy(train[0])[:, 0:n]

    for step_ahead in range(n_periods):
        y_pred_one = model.predict(data_pred[:, step_ahead:data_pred.shape[1]])
        y_pred_one = y_pred_one[..., np.newaxis]
        data_pred = np.concatenate([data_pred, y_pred_one], axis = 1)
    series_pred = scale(np.copy(data_pred), values)
    return(series_pred)
    
    
def rescale(dataset, n=1):
    """Perform a standard rescaling of the data
    
    Parameters
    ----------
    dataset : np.array
        An array containing the model data.
    n : int
        Is the degree of the polynomial to subtract 
        the slope. By default it is set to `1`.
        
    Returns
    -------
    data_scaled : np.array
        An array containing the scaled data.
        
    mu : np.array
        An array containing the mean of the 
        original data.
    sigma : np.array
        An array containing the standard 
        deviation of the original data.
    """
    
    mu = []
    sigma = []
    fitting = []
    
    try:
        xaxis = range(dataset.shape[1])
    except:
        error_type = 'IndexError'
        msg = 'Trying to access an item at an invalid index.'
        print(f'{error_type}: {msg}')
        return None
    for i in range(dataset.shape[0]):
        if n != None:
            fit = np.polyfit(xaxis, dataset[i, :, 0], n)
            f = np.poly1d(fit)
            poly = f(xaxis)
            fitting.append(f)
        else:
            fitting.append(0.0)
        dataset[i, :, 0] += -poly
        mu.append(np.min(dataset[i, :, 0]))
        if np.max(dataset[i, :, 0]) != 0: 
            sigma.append(np.max(dataset[i, :, 0])-mu[i])
        else:
            sigma.append(1)
            
        dataset[i, :, 0] = 2*((dataset[i, :, 0] - mu[i]) / sigma[i])-1
         
    values = [mu, sigma, fitting]
    
    return dataset, values
 
def scale(dataset, values):
    """Performs the inverse operation to the rescale function
    
    Parameters
    ----------
    dataset : np.array
        An array containing the scaled data.
    values : np.ndarray
        A set of values returned by the rescale function.
    
    """
    
    for i in range(dataset.shape[0]):
        dataset[i, :, 0] += 1
        dataset[i, :, 0] /= 2
        dataset[i, :, 0] = dataset[i, :, 0]*values[1][i]
        dataset[i, :, 0] += values[0][i]
        dataset[i, :, 0] += values[2][i](range(dataset.shape[1]))
    
    return dataset