import numpy as np
from numpy import linalg as la
import LinearModelsWeek5 as lm


def system_2sls(y: list, x: list, z: list) -> dict:
    """Takes a list of the y, x and z arrays that we perform a system 2SLS
    regression on. The last element of y and x list be the array with all
    observations.

    Args:
        >> y (list): A list of arrays, where each array are all individuals
        for one time period. The last array should be all individuals for 
        all time periods.
        >>x (list): A list of arrays, where each array are all individuals
        for one time period. The last array should be all individuals for 
        all time periods.
        z (list): A list of arrays, where each array are all individuals
        for one time period. This should not have an extra array with all 
        individuals for all time periods.

    Returns:
        dict: Returns a dictionary witht he results from S2SLS.
    """
    # Intialize som helper variables
    n_rows = z[0].shape[0]
    n_cols = len(z)

    # Initialize the arrays to fill from first_stage loop
    x_predicted = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        x_predicted[:, i] = first_stage(y[i], x[i], z[i])
    
    # Reshape the x_predicted into a single column
    x_predicted = None # FILL IN

    b_hat = None  # Estimate the second step
    residual = None  # Calculate residuals (use x and not xhat)
    SSR = None  # Calculate SSR
    sigma2 = None  # Calculate sigma2 With 1/(n-k) adjustment
    cov = None  # Calculate covariance
    se = None # Calculate standard errors
    t_values = None # Calculate t-values
    
    results = [b_hat, se, sigma2, t_values, np.array(0)]
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2']
    return dict(zip(names, results))


def first_stage(y: np.array, x: np.array, z: np.array) -> np.array:
    """Perform the first stage regression of x on z. Then uses y to create
    the predictions x-hat

    Args:
        >> y (np.array): An array that have all individuals, but for one time 
        period only.
        >> x (np.array): An array that have all individuals, but for one time 
        period only.
        >> z (np.array): An array that have all individuals, but for one time 
        period only.

    Returns:
        np.array: The predicted x-hats.
    """
    # Estimate the first stage regression, and return predicted values.
    pass


def sequential_instruments(x:np.array, T:int) -> np.array:
    """Takes x, and creates the instrument matrix.

    Args:
        >> x (np.array): The instrument vector that we will use to create a new
        instrument matrix that uses all possible instruments each period.
        >> T (int): Number of periods (in the original dataset, before removing
        observations to do first differences or lags). 

    Returns:
        np.array: A (n*(T - 1), k*T*(T - 1)/2) matrix, that has for each individual
        have used all instruments available each time period.
    """

    # Create some helper variables
    n = int(x.shape[0]/(T - 1))
    k = x.shape[1]
    
    # Initialize the Z matrix, that we will fill up using the loop.
    Z = np.zeros((n*(T - 1), int(k*T*(T - 1) / 2)))

    # I recommend using two loops:
    # First loops over the persons.
        # It can be a good idea to make a temporary variable, where you retreive
        # the observations we have for that person, that makes it easier in the
        # next loop.
    
    # Second loops over the time periods for that person.
        # Here you should therefore for the first time period get only one
        # instrument. Then for the second time period get two instruments. etc.
    
    
    # So for each person, you should get a matrix that looks like Zo from the
    # Part 3 text. You should then stack these on top of each other, so that
    # so that you get a very tall matrix.
    
    return Z