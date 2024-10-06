# Simulate data.

import numpy as np
from scipy import stats

def get_iden(T):
    """
    Generate a T*T identity matrix.

    Parameters
    ----------
    T: int; pos. Number of time periods.

    Returns
    -------
    iden: array(T,T). T*T identity matrix.

    """
    iden = np.eye(T)
    return iden

def get_cor_ar1(T, r):
    """
    Autoregressive correlation: Generate a T*T correlation matrix per AR(1).

    Parameters
    ----------
    T: int. Number of time periods.
    r: float, list/tuple/array(T-1,), in [-1,1].
        Correlation coefficient between two adjacent periods.
        If scalar, AR(1) is stationary.
        If vector, AR(1) is non-stationary.

    Returns
    -------
    cor: array(T,T). Correlation matrix.
    
    """
    if isinstance(r, (float, int)):
        r = r * np.ones(T-1)
    elif isinstance(r, (list, tuple)):
        assert len(r) == T-1, 'len(r) != T-1.'
        r = np.asarray(r)
    elif isinstance(r, np.ndarray):
        assert r.ndim == 1, 'r.ndim != 1.'
        assert r.size == T-1, 'r.size != T-1.'

    r0 = 1.
    cor = r0 * np.eye(T)
    
    for t in range(1, T):
        r0 *= r[t-1]
        cor += r0 * (np.eye(T, k=t) + np.eye(T, k=-t))

    return cor

def cor2cov(cor, scale):
    """
    Convert correlation matrix to covariance matrix.

    Parameters
    ----------
    cor: array(T,T). Correlation matrix.
    scale: float/int, list/tuple/array(T,). Standard deviation scalar or vector.

    Returns
    -------
    cov: array(T,T). Covariance matrix.
    
    """
    assert cor.ndim == 2, '[cor] is not a matrix.'
    assert cor.shape[0] == cor.shape[1], '[cor] is not a square matrix.'
    T = cor.shape[0]
    
    if isinstance(scale, (float, int)):
        scale_diag = scale * np.eye(T) # (T,T)
    elif isinstance(scale, (list, tuple)):
        assert len(scale) == T, 'len(scale) != T.'
        scale_diag = np.diag(np.asarray(scale)) # (T,T)
    elif isinstance(scale, np.ndarray):
        assert scale.shape == (T,), 'scale.shape != (T,).'
        scale_diag = np.diag(scale) # (T,T)
    else:
        raise TypeError('Type of argument [scale] not supported.')
    
    cov = scale_diag @ cor @ scale_diag
    return cov

def cov2cor(cov, returns=None):
    """
    Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    cov: array(T,T). Covariance matrix.
    returns: str. See below.

    Returns
    -------
    cor: array(T,T). Correlation matrix.
    scale: array(T,), if returns='scale'. The standard deviation vector.
    scale_diag: array(T,T), if returns='scale_diag'. The standard deviation diagonal matrix.
    
    """
    assert cov.ndim == 2, '[cov] is not a matrix.'
    assert cov.shape[0] == cov.shape[1], '[cov] is not a square matrix.'

    scale = np.sqrt(np.diag(cov)) # (T,)
    scale_diag = np.diag(scale) # (T,T)
    scale_diag_inv = np.diag(1/scale) # (T,T)
    cor = scale_diag_inv @ cov @ scale_diag_inv # (T,T)

    if returns == 'scale':
        return cor, scale
    elif returns == 'scale_diag':
        return cor, scale
    elif returns == 'scale, scale_diag':
        return cor, scale, scale_diag
    else:
        return cor
    
class mv_tdis():

    def __init__(self, loc, cov, df, cov_type='mean'):
        """
        Inverse Wishart distribution.

        Parameters
        ----------
        cov: array(T,T). The 'center' of inv-Wishart distribution.
        df: int. Degrees of freedom.
        cov_type: str. If 'mean', cov is the mean; if 'mode', cov is the mode;
            if 'natural', cov is the scale matrix divided by df (following common inv-W).
        
        Notes
        -----
        The parameterization of cov may be different.
        
        """
        self.cov, self.df, self.cov_type = np.asarray(cov), df, cov_type
        assert self.cov.ndim == 2, 'cov.ndim != 2'
        assert self.cov.shape[0] == self.cov.shape[1], 'cov is not a square matrix.'        
        self.T = self.cov.shape[0]
        assert self.df - self.T-1 > 0, 'df-T-1 <= 0.'

        if self.cov_type == 'mean':
            self.distr = stats.invwishart(df=self.df, scale=(self.df-self.T-1)*self.cov)
        elif self.cov_type == 'mode':
            self.distr = stats.invwishart(df=self.df, scale=(self.df+self.T+1)*self.cov)
        elif self.cov_type == 'natural':
            self.distr = stats.invwishart(df=self.df, scale=self.df*self.cov)
        else:
            raise ValueError("Argument [cov_type] should be ['mean','mode','natural'].")

    def draw(self, size=1, rng=None):
        """
        Draw random samples.

        Parameters
        ----------
        size: int, array. Number of samples to draw.

        Return
        samples: if size=1, return array(T,T);
            if size has shape (a,b), return array(a,b,T,T)
        """
        return self.distr.rvs(size=size, random_state=rng)
    
class inv_wishart():

    def __init__(self, cov, df, cov_type='mean'):
        """
        Inverse Wishart distribution.

        Parameters
        ----------
        cov: array(T,T). The 'center' of inv-Wishart distribution.
        df: int. Degrees of freedom.
        cov_type: str. If 'mean', cov is the mean; if 'mode', cov is the mode;
            if 'natural', cov is the scale matrix divided by df (following common inv-W).
        
        Notes
        -----
        The parameterization of cov may be different.
        
        """
        self.cov, self.df, self.cov_type = np.asarray(cov), df, cov_type
        assert self.cov.ndim == 2, 'cov.ndim != 2'
        assert self.cov.shape[0] == self.cov.shape[1], 'cov is not a square matrix.'        
        self.T = self.cov.shape[0]
        assert self.df - self.T-1 > 0, 'df-T-1 <= 0.'

        if self.cov_type == 'mean':
            self.distr = stats.invwishart(df=self.df, scale=(self.df-self.T-1)*self.cov)
        elif self.cov_type == 'mode':
            self.distr = stats.invwishart(df=self.df, scale=(self.df+self.T+1)*self.cov)
        elif self.cov_type == 'natural':
            self.distr = stats.invwishart(df=self.df, scale=self.df*self.cov)
        else:
            raise ValueError("Argument [cov_type] should be ['mean','mode','natural'].")

    def draw(self, size=1, rng=None):
        """
        Draw random samples.

        Parameters
        ----------
        size: int, array. Number of samples to draw.

        Return
        samples: if size=1, return array(T,T);
            if size has shape (a,b), return array(a,b,T,T)
        """
        return self.distr.rvs(size=size, random_state=rng)

class nonnorm():

    def __init__(self, loc, scale, skew, kurt, tol=1e-6, maxiter=100):
        """
        Simulate univariate non-normal data using the Fleishman (1978) approach.

        Parameters
        ----------
        loc: float. Location.
        scale: float. Scale.
        skew: float. Skewness (standard).
        kurt: float. Kurtosis (standard).

        Notes
        -----
        For standard normal distribution: 0,1,0,3
        test: rv = nonnorm(10,5,1,3)
        
        """

        self.loc, self.scale, self.skew, self.kurt = loc, scale, skew, kurt
        self.tol, self.maxiter = tol, maxiter

        assert self.kurt >= self.skew**2 + 1, f'kurt < skew**2+1.'

        # Solve polynomial parameters. 
        par = mv_newton(f=f_Fleishman, \
                        x_init=np.array([1.,0.,0.]), \
                        f_kwargs=dict(sk=self.skew, ku=self.kurt), \
                        tol=self.tol, maxiter=self.maxiter, disp=True)
        self.b, self.c, self.d = par
        self.a = - self.c

    def draw(self, size=1, rng=None):
        """
        Draw random samples.

        Parameters
        ----------
        size: int, tuple. Number of samples to draw.

        Returns
        -------
        samples: if size is int, return array(size,);
            if size is tuple (a,b), return array(a,b).
        """
        if rng is None:
            x = np.random.default_rng().normal(loc=0., scale=1., size=size) # (size)
        else:
            x = rng.normal(loc=0., scale=1., size=size) # (size)
        y = self.a + self.b * x + self.c * (x**2) + self.d * (x**3) # (size)
        y = self.loc + self.scale * y # (size)
        
        return y
    
class mv_nonnorm():

    def __init__(self, cor, loc, scale, skew, kurt, tol=1e-6, maxiter=100):
        """
        Simulate multivariate non-normal data using the Vale and Maurelli (1983) approach.
        This is a python version of the [rValeMaurelli] function in R package [SimDesign].
        Special thanks to the author Phil Chalmers.

        Parameters
        ----------
        cor: array(T,T). Correlation matrix.
        loc: array(T,). Location vector of marginal univariate ones.
        scale: array(T,). Scale vector of marginal univariate ones.
        skew: array(T,). Skewness vector of marginal univariate ones (standard).
        kurt: array(T,). Kurtosis vector of marginal univariate ones (standard).

        Notes
        -----
        test: rv = mv_nonnorm([[1,0.75],[0.75,1]], [0,0], [1,1], [1,1], [4,4])
        
        """
        self.cor, self.loc, self.scale, self.skew, self.kurt = \
        np.asarray(cor), np.asarray(loc), np.asarray(scale), \
        np.asarray(skew), np.asarray(kurt)
        self.tol, self.maxiter = tol, maxiter
        assert self.cor.ndim == 2
        assert self.loc.ndim == self.scale.ndim == self.skew.ndim == self.kurt.ndim == 1
        assert self.cor.shape[0] == self.cor.shape[1] == self.loc.shape[0] == \
               self.scale.shape[0] == self.skew.shape[0] == self.kurt.shape[0]
        self.cov = cor2cov(self.cor, self.scale) # (T,T)
        self.T = self.cor.shape[0] # scalar
               
        for t in range(self.T):
            assert self.kurt[t] >= self.skew[t]**2 + 1, \
                   f'kurt < skew**2+1 at the {t+1} th dimension.'

        # Solve polynomial parameters.
        self.b, self.c, self.d = [], [], []
        
        for t in range(self.T):
            par = mv_newton(f=f_Fleishman, \
                            x_init=np.array([1.,0.,0.]), \
                            f_kwargs=dict(sk=self.skew[t], ku=self.kurt[t]), \
                            tol=self.tol, maxiter=self.maxiter, disp=False)
            self.b.append(par[0])
            self.c.append(par[1])
            self.d.append(par[2])

        self.b, self.c, self.d = np.asarray(self.b), np.asarray(self.c), np.asarray(self.d)
        self.a = - self.c # (T,)

        # Solve correlation parameters.
        self.cor_mvn = np.diag(np.ones(self.T)) # (T,T)

        for i in range(self.T-1):
            for j in range(i+1, self.T):               
                bi, bj = self.b[i], self.b[j] # scalar
                ci, cj = self.c[i], self.c[j] # scalar
                di, dj = self.d[i], self.d[j] # scalar
                p0 = 6*di*dj # scalar
                p1 = 2*ci*cj # scalar
                p2 = bi*bj + 3*bi*dj + 3*bj*di + 9*di*dj # scalar
                p3 = - self.cor[i,j] # scalar
                roots = np.roots([p0,p1,p2,p3]) # (3,)
                roots = roots[np.isreal(roots)]
                roots = roots[(roots>=-1)&(roots<=1)]
                assert roots.shape == (1,), \
                       'Roots of the cubic equation do not satisfy regularity conditions.'
                rij_mvn = roots[0].real # scalar
                self.cor_mvn[i,j] = rij_mvn
                self.cor_mvn[j,i] = rij_mvn       
                
    def draw(self, size=1, rng=None):
        """
        Draw random samples.

        Parameters
        ----------
        size: int, tuple. Number of samples to draw.

        Returns
        -------
        samples: if size is int, return array(size, T);
            if size is tuple (a,b), return array(a,b,T).
        """
        if rng is None:
            x = np.random.default_rng().multivariate_normal(\
                mean=np.zeros(self.T), \
                cov=self.cor_mvn, \
                size=size) # (size, T)
        else:
            x = rng.multivariate_normal(\
                mean=np.zeros(self.T), \
                cov=self.cor_mvn, \
                size=size) # (size, T)
        if isinstance(size, int):
            y = np.expand_dims(self.a, 0) + \
                np.expand_dims(self.b, 0) * x + \
                np.expand_dims(self.c, 0) * (x**2) + \
                np.expand_dims(self.d, 0) * (x**3) # (size, T)
            y = np.expand_dims(self.loc, 0) + \
                np.expand_dims(self.scale, 0) * y # (size, T)
        elif isinstance(size, tuple):
            add_axis = tuple(range(len(size)))
            y = np.expand_dims(self.a, add_axis) + \
                np.expand_dims(self.b, add_axis) * x + \
                np.expand_dims(self.c, add_axis) * (x**2) + \
                np.expand_dims(self.d, add_axis) * (x**3) # (size, T)
            y = np.expand_dims(self.loc, add_axis) + \
                np.expand_dims(self.scale, add_axis) * y # (size, T)
        else:
            raise TypeError('Type of [size] not supported.')

        return y  

def f_Fleishman(x, sk, ku):
    """
    Parameters
    ----------
    x: array(3,). Vector [b,c,d] in the Fleishman polynomial.
    sk: float. Desired univariate skewness.
    ku: float. Desired univariate kurtosis.

    Returns
    -------
    y: array(3,). Rhs of Fleishman equations.
    dy: array(3,3). 3*3 Jacobian matrix J with J_ij = dy_i/dx_j. 

    Notes
    -----
    a = -c
    
    """
    b, c, d = x # scalars
    y1 = b**2 + 6*b*d + 2*(c**2) + 15*(d**2) - 1
    y2 = 2*c * (b**2 + 24*b*d + 105*(d**2) + 2) - sk
    y3 = 24 * (b*d + (c**2)*(1+b**2+28*b*d) + (d**2)*(12+48*b*d+141*(c**2)+225*(d**2))) - ku + 3

    dy1 = [2*b+6*d, 4*c, 6*b+30*d]
    dy2 = [2*c*(2*b+24*d), 2*(b**2+24*b*d+105*(d**2)+2), 2*c*(24*b+210*d)]
    dy3 = [24*(d+(c**2)*(2*b+28*d)+48*(d**3)), \
           24*(2*c*(1+b**2+28*b*d)+282*c*(d**2)), \
           24*(b+28*b*(c**2)+24*d+144*b*(d**2)+282*(c**2)*d+900*(d**3))]
    
    return np.array([y1, y2, y3]), np.array([dy1,dy2,dy3])

def mv_newton(f, x_init, f_kwargs={}, tol=1e-6, maxiter=100, disp=True):
    """
    Multivariate Newton-Raphson method.

    Parameters
    ----------
    f: func(x, **f_kwargs), (k,) -> (k,), (k,k).
        A function y=f(x) with [input: parameter]; [output: function, gradient].
    x_init: array(k,). Initial guess.
    f_kwargs: tuple. Additional keyword arguments passed to function [f].

    Returns:
    -------
    x: array(k,). Approximate solution of y(x)=0.
    
    """
    x_old = x_init.copy() # (k,)
    
    for i in range(maxiter+1):
        f_x, df_x = f(x_old, **f_kwargs)
        x_new = x_old - np.linalg.inv(df_x) @ f_x # (k,)                
        err_abs = np.abs(x_new - x_old) # (k,)
        if np.amax(err_abs) < tol:
            break
        else:
            x_old = x_new.copy()

    if i == maxiter:
        print(f'Convergence not reached with [tol]={tol} and [maxiter]={maxiter}.')        

    if disp:
        print(f'Number of iterations = {i+1}.')    
        print(f'Absoulte error = {err_abs}.')

    return x_new
