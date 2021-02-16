import numpy as np
from numpy import linalg as la
from tabulate import tabulate



def estimate( 
        y: np.array, x: np.array, transform='', n=None, t=None
    ) -> dict:
    """Takes some np.arrays and estimates regular OLS, FE or FD.
    

    Args:
        y (np.array): The dependent variable, needs to have the shape (n*t, 1)
        x (np.array): The independent variable(s). If only one independent 
        variable, then it needs to have the shape (n*t, 1).
        transform (str, optional): Specify if estimating fe or fd, in order 
        to get correct variance estimation. Defaults to ''.
        n (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        t (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Returns:
        dict: A dictionary with the results from the ols-estimation.
    """
    
    b_hat = None # Fill in
    resid = None # Fill in
    u_hat = None # Fill in
    SSR = None # Fill in
    SST = None # Fill in
    R2 = None # Fill in

    sigma, cov, se = variance(transform, SSR, x, n, t)
    t_values =  None # Fill in
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.array, x: np.array) -> np.array:
    """Estimates OLS using input arguments.

    Args:
        y (np.array): Check estimate()
        x (np.array): Check estimate()

    Returns:
        np.array: Estimated beta hats.
    """
    return   # Fill in

def variance( 
        transform: str, 
        SSR: float, 
        x: np.array, 
        n: int,
        t: int
    ) -> tuple :
    """Use SSR and x array to calculate different variation of the variance.

    Args:
        transform (str): Specifiec if the data is transformed in any way.
        SSR (float): SSR
        x (np.array): Array of independent variables.
        n (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        t (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        tuple: [description]
    """
    if not transform:
          sigma = None # Fill in
    elif transform.lower() == 'fe':
          sigma = None # Fill in
    elif transform.lower() in ('be', 're'):
          sigma = None # Fill in
    else:
        raise Exception('Invalid transform provided.')
    
    cov =  None # Fill in
    se =  None # Fill in
    return sigma, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        **kwargs
    ) -> None:
    label_y, label_x = labels
    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma').item():.3f}")


def perm( Q_T: np.array, A: np.array, t=0) -> np.array:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z