import numpy as np

# Function to propagate the state forward in time
def xf_i1(xa_i, M, Q):  ## Model no no
    """
    Update forecast state using model evolution matrix and process noise.
    :param xa_i: Current analysis ensemble (K members, d dimensions)
    :param M: Model evolution matrix (d x d)
    :param Q: Process noise covariance matrix (d x d)
    :return: Forecast ensemble (K members, d dimensions)
    """
    ensemble_f = np.dot(M, xa_i.T).T  # Apply model evolution
    return ensemble_f + np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, size=xa_i.shape[0])

# Function to update a single ensemble member during the analysis step
def xa_k(xf_k, K_gain, yo_k): #kill H
    """
    Update a single ensemble member using Kalman gain and perturbed observation.
    :param xf_k: Forecast state for the k-th ensemble member (d dimensions)
    :param K_gain: Kalman gain (d x p)
    :param yo_k: Perturbed observation for the k-th ensemble member (p dimensions)
    :return: Analysis state for the k-th ensemble member (d dimensions)
    """
    return xf_k + K_gain @ (yo_k - xf_k)

# Function to compute the Kalman gain
def kalman_gain(P_f, R): #kill H
    """
    Compute the Kalman gain matrix.
    :param P_f: Forecast error covariance matrix (d x d)
    :param R: Observation error covariance matrix (p x p)
    :return: Kalman gain matrix (d x p)
    """
    return P_f @ np.linalg.inv(P_f+ R)

# Function to compute the covariance matrix of an ensemble
def covariance(ensemble): #Pf_i
    """
    Compute the covariance matrix of an ensemble.
    :param ensemble: Ensemble matrix (K members, d dimensions)
    :return: Covariance matrix (d x d)
    """
    mean = np.mean(ensemble, axis=0)
    return (ensemble - mean).T @ (ensemble - mean) / (ensemble.shape[0] - 1)

# Main driver code
def po_enkf_step(xa_i, M, Q, H, R, y_o, K): ## Model nono
    """
    Perform one step of PO-EnKF.
    :param xa_i: Initial analysis ensemble (K members, d dimensions)
    :param M: Model evolution matrix (d x d)
    :param Q: Process noise covariance matrix (d x d)
    :param H: Observation operator (p x d)
    :param R: Observation error covariance matrix (p x p)
    :param y_o: Observed data (p dimensions)
    :param K: Number of ensemble members
    :return: Analysis ensemble (K members, d dimensions)
    """
    # Forecast step
    xf_ensemble = xf_i1(xa_i, M, Q)
    P_f = covariance(xf_ensemble)  # Forecast error covariance
    K_gain = kalman_gain(P_f, H, R)  # Kalman gain

    # Analysis step
    xa_ensemble = np.zeros_like(xf_ensemble)
    for k in range(K):
        yo_k = y_o + np.random.multivariate_normal(np.zeros(R.shape[0]), R)  # Perturb observation
        xa_ensemble[k] = xa_k(xf_ensemble[k], K_gain, yo_k, H)

    return xa_ensemble

