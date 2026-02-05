import numpy as np

def verify_parseval_3d(arr, rtol=1e-12, atol=1e-12):
    """
    Numerically verify Parseval's theorem for a 3D array.

    Parameters
    ----------
    arr : ndarray
        Real or complex 3D array
    rtol, atol : float
        Tolerances for numerical comparison

    Returns
    -------
    lhs : float
        Sum of |f(x)|^2 in real space
    rhs : float
        (1/N) sum of |F(k)|^2 in Fourier space
    """
    if arr.ndim != 3:
        raise ValueError("Input array must be 3D")

    # Real-space energy
    lhs = np.sum(np.abs(arr)**2)

    # Fourier transform
    F = np.fft.fftn(arr)

    # Total number of grid points
    N = arr.size

    # Fourier-space energy
    rhs = np.sum(np.abs(F)**2) / N

    print(f"Real-space energy   = {lhs:.15e}")
    print(f"Fourier-space energy= {rhs:.15e}")
    print(f"Relative error      = {abs(lhs-rhs)/lhs:.3e}")

    assert np.allclose(lhs, rhs, rtol=rtol, atol=atol), \
        "Parseval's theorem failed!"

    return lhs, rhs

verify_parseval_3d(np.random.rand(16,16,32))