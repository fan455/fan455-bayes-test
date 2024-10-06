use fan455_arrf64::*;
use std::iter::zip;
use super::bayes_stat::*;


#[allow(non_snake_case)]
pub struct FreqsPlrBase {
    pub y: Arr<f64>, // (N*T, 1)
    pub X: Mat<f64>, // (N*T, K)
    pub T: usize,
    pub N: usize,
    pub K: usize,
    pub beta: Arr<f64>,
    pub beta_cov: Mat<f64>,
    pub u: Mat<f64>,
    pub CI: f64,
}


impl FreqsPlrBase {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr<f64>, X: Mat<f64>, T: usize ) -> Result<Self, &'static str> {
        if y.size() % T != 0 {
            return Err("The values of T and N are incorrect.");
        }
        let N: usize = y.size() / T;
        let K: usize = X.ncol();
        if K < T {
            return Err("K >= T is needed for the PLR model.");
        }
        if N < K {
            return Err("N >= K is needed for the PLR model.");
        }
        let beta = Arr::<f64>::new(K);
        let beta_cov = Mat::<f64>::new(K, K);
        let u = Mat::<f64>::new(N, T);
        let CI: f64 = 0.95;
        Ok( Self{ y, X, T, N, K, beta, beta_cov, u, CI } )
    }

    #[inline]
    pub fn get_u( &mut self ) {
        self.u.copy(&self.y);
        dgemv(-1., &self.X, &self.beta, 1., &mut self.u, NO_TRANS);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn t_test( &self, k: usize, H0: f64 ) -> (f64, f64) {
        t_test_twotail(
            self.beta[k], H0, (self.N*self.T-self.K) as f64, self.beta_cov[(k, k)].sqrt()
        )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn z_test( &self, k: usize, H0: f64 ) -> (f64, f64) {
        z_test_twotail(self.beta[k], H0, self.beta_cov[(k, k)].sqrt())
    }

    #[inline] #[allow(non_snake_case)]
    pub fn t_test_ci( &self, k: usize ) -> (f64, f64) {
        t_test_ci(
            self.beta[k], self.CI, (self.N*self.T-self.K) as f64, self.beta_cov[(k, k)].sqrt()
        )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn z_test_ci( &self, k: usize ) -> (f64, f64) {
        z_test_ci(
            self.beta[k], self.CI, self.beta_cov[(k, k)].sqrt()
        )
    }

    #[inline]
    pub fn reset( &mut self ) {
        self.beta.reset();
        self.beta_cov.reset();
    }
}


pub struct PlrOls {
    pub base: FreqsPlrBase,
    pub sigma2: f64
}

impl PlrOls {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr<f64>, X: Mat<f64>, T: usize ) -> Result<Self, &'static str> {
        let base: FreqsPlrBase = FreqsPlrBase::new(y, X, T)?;
        let sigma2: f64 = 0.;
        Ok( Self{ base, sigma2 } )
    }

    #[inline]
    pub fn estimate( &mut self ) {
        self.get_beta();
        self.base.get_u();
        self.get_sigma2();
        self.get_beta_cov();
    }

    #[inline]  #[allow(non_snake_case)]
    pub fn estimate_sandwich( &mut self ) {
        self.get_beta();
        self.base.get_u();
        self.get_sigma2();
        self.get_beta_cov();
        let mut V = Mat::<f64>::new(self.base.T, self.base.T);
        self.get_V(&mut V);
        self.get_beta_cov_sandwich(&mut V);
    }

    #[inline]
    pub fn get_beta( &mut self ) {
        dsyrk(1., &self.base.X, 0., &mut self.base.beta_cov, TRANS, LOWER); // Get beta_icov = X^T*X
        dgemv(1., &self.base.X, &self.base.y, 0., &mut self.base.beta, TRANS); // Get beta <- X^T * y
        dposv(&mut self.base.beta_cov, &mut self.base.beta, LOWER); // Get beta_icov_lo and solve beta.
    }

    #[inline]
    pub fn get_sigma2( &mut self ) {
        self.sigma2 = self.base.u.sumsquare() / (self.base.N * self.base.T - self.base.K) as f64; 
    }

    #[inline]
    pub fn get_beta_cov( &mut self ) {
        dpotri(&mut self.base.beta_cov, LOWER); // Get beta_cov.
        self.base.beta_cov.scale(self.sigma2);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_V( &self, V: &mut Mat<f64> ) {
        dsyrk(1./self.base.N as f64, &self.base.u, 0., V, TRANS, LOWER);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_beta_cov_sandwich( &mut self, V0: &mut Mat<f64> ) {
        dpotrf(V0, LOWER);
   
        let mut Xi = Mat::<f64>::new(self.base.T, self.base.K);
        let A = self.base.beta_cov.clone();
        self.base.beta_cov.reset();
        let mut AB = Mat::<f64>::new(self.base.K, self.base.K);
        let mut sel: Vec<usize> = vec![0; self.base.T];

        for i in 0..self.base.N {
            for (sel_, t) in zip(sel.as_mut_slice(), 0..self.base.T) {
                *sel_ = i + t * self.base.N;
            }
            Xi.get_rows(&self.base.X, sel.as_slice());
            dtrmm(1./self.sigma2, V0, &mut Xi, LEFT, TRANS, LOWER);
            dsyrk(1., &Xi, 1., &mut self.base.beta_cov, TRANS, LOWER);
        }
        self.base.beta_cov.copy_lower_to_upper();
        dsymm(1., &A, &self.base.beta_cov, 0., &mut AB, LEFT, LOWER);
        dsymm(1., &A, &AB, 0., &mut self.base.beta_cov, RIGHT, LOWER);
    }
}


#[allow(non_snake_case)]
pub struct PlrGls {
    pub base: FreqsPlrBase,
    pub V: Mat<f64>,
    pub V_lo: Mat<f64>,
    pub V_inv: Mat<f64>,
    pub V_invlo: Mat<f64>
}

impl PlrGls {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr<f64>, X: Mat<f64>, T: usize, V: Mat<f64> ) -> Result<Self, &'static str> {
        let base: FreqsPlrBase = FreqsPlrBase::new(y, X, T)?;
        let mut V_lo = V.clone();
        dpotrf(&mut V_lo, LOWER);
        let mut V_inv = V_lo.clone();
        dpotri(&mut V_inv, LOWER);
        let mut V_invlo = V_inv.clone();
        dpotrf(&mut V_invlo, LOWER);
        Ok( Self{ base, V, V_lo, V_inv, V_invlo } )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn new_feasible( y: Arr<f64>, X: Mat<f64>, T: usize ) -> Result<Self, &'static str> {
        let mut m: PlrOls = PlrOls::new(y, X, T)?;
        m.get_beta();
        m.base.get_u();
        let mut V = Mat::<f64>::new(m.base.T, m.base.T);
        m.get_V(&mut V);
        Ok( PlrGls::new( m.base.y, m.base.X, T, V )? )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn new_uninit( y: Arr<f64>, X: Mat<f64>, T: usize ) -> Result<Self, &'static str> {
        let base: FreqsPlrBase = FreqsPlrBase::new(y, X, T)?;
        let V = Mat::<f64>::new(T, T);
        let V_lo = Mat::<f64>::new(T, T);
        let V_inv = Mat::<f64>::new(T, T);
        let V_invlo = Mat::<f64>::new(T, T);
        Ok( Self{ base, V, V_lo, V_inv, V_invlo } )
    }

    #[inline]
    pub fn estimate( &mut self ) {
        self.get_beta();
        self.get_beta_cov();
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_V( mut self ) -> Self {
        let mut m: PlrOls = PlrOls::new(self.base.y, self.base.X, self.base.T).unwrap();
        m.get_beta();
        m.base.get_u();
        m.get_V(&mut self.V);
        self.base.y = m.base.y;
        self.base.X = m.base.X;
        self
    }

    #[inline] #[allow(non_snake_case)]
    pub fn solve_V( &mut self ) {
        self.V_lo.copy(&self.V);
        dpotrf(&mut self.V_lo, LOWER);

        self.V_inv.copy(&self.V_lo);
        dpotri(&mut self.V_inv, LOWER);

        self.V_invlo.copy(&self.V_inv);
        dpotrf(&mut self.V_invlo, LOWER);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_beta( &mut self ) {
        // Compute beta_icov and X^T * iV * y.
        let mut yi = Arr::<f64>::new(self.base.T);
        let mut Xi = Mat::<f64>::new(self.base.T, self.base.K);
        let mut V_inv_yi = Arr::<f64>::new(self.base.T);
        let mut sel: Vec<usize> = vec![0; self.base.T];

        for i in 0..self.base.N {
            for (sel_, t) in zip(sel.as_mut_slice(), 0..self.base.T) {
                *sel_ = i + t * self.base.N;
            }
            yi.get_elems(&self.base.y, sel.as_slice());
            Xi.get_rows(&self.base.X, sel.as_slice());

            dsymv(1., &self.V_inv, &yi, 0., &mut V_inv_yi, LOWER);
            dgemv(1., &Xi, &V_inv_yi, 1., &mut self.base.beta, TRANS);

            dtrmm(1., &self.V_invlo, &mut Xi, LEFT, TRANS, LOWER);
            dsyrk(1., &Xi, 1., &mut self.base.beta_cov, TRANS, LOWER); // It is actually beta_icov now.
        }
        // Compute beta.
        dposv(&mut self.base.beta_cov, &mut self.base.beta, LOWER); // It is actually beta_icov_L now. And beta is solved.
    }

    #[inline]
    pub fn get_beta_cov( &mut self ) {
        dpotri(&mut self.base.beta_cov, LOWER); // Get beta_cov.
    }

}
