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
    x_predicted = x_predicted.reshape(-1, 1)

    b_hat = lm.est_ols(y[-1], x_predicted)
    residual = y[-1] - x[-1] @ b_hat
    SSR = residual.T @ residual
    sigma2 = SSR/(y[-1].size - x[-1].shape[1])
    cov = sigma2 * la.inv(x[-1].T @ x[-1])
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    t_values = b_hat/se
    
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


def est_gmm(
        W: np.array, y: np.array, x: np.array, z: np.array, 
        t: int, step=1
    ) -> np.array:
    
    n = int(y.size/t)
    k = x.shape[1]
    s = 0
    while s<step:
        b_hat = la.inv(x.T@z@W@z.T@x)@x.T@z@W@z.T@y
        residual = y - x@b_hat

        if s==0:
            cov, se = lm.robust(z@W@z.T@x, residual, t)
        else:
            se = np.sqrt(
                la.inv(x.T@z@W@z.T@x).diagonal()
                ).reshape(-1, 1)

        W = np.zeros((W.shape[0], W.shape[1]))
        for i in range(n):
            slice_ob = slice(i*t, (i + 1)*t)
            zi = z[slice_ob]
            resi = residual[slice_ob]
            W += zi.T@(resi@resi.T)@zi
        W = la.inv(W)
        J = residual.T@ z@W@z.T @residual
        sigma2 = (residual.T @ residual)/(t*n - k)
        t_values = b_hat/se
        s += 1

    results = [b_hat, se, sigma2, t_values, np.array(0), J]
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'J']
    return dict(zip(names, results))


def sequential_instruments(x:np.array, T:int):
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

    n = int(x.shape[0]/(T - 1))
    k = x.shape[1]
    Z = np.zeros((n*(T - 1), int(k*T*(T - 1) / 2)))

    # Loop through all persons, and then loop through their time periods.
    # If first time period, use only that as an instrument.
    # Second time period, use the first and this time period as instrument, etc. 
    # Second last time period (T-1)

    # Loop over each individual, we take T-1 steps.
    for i in range(0, n*(T - 1), T - 1):
        # We make some temporary arrays for the current individual
        zi = np.zeros((int(k*T*(T - 1) / 2), T - 1))
        xi = x[i: i + T - 1]

        # j is a help variable on how many instruments we create each period.
        # The first period have 1 iv variable, the next have 2, etc.
        j = 0
        for t in range(1, T):
            zi[j: (j + t), t - 1] = xi[:t].reshape(-1, )
            j += t
        # It was easier to fill the instruments row wise, so we need to transpose
        # the individual matrix before we add it to the main matrix.
        Z[i: i + T - 1] = zi.T
    return Z