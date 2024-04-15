use fan455_arrf64::*;
use fan455_arrf64_macro::*;
use statrs::distribution::ContinuousCDF;
use std::iter::zip;


#[allow(non_snake_case)]
pub struct PlrBase {
    pub y: Arr1<f64>, // (N*T, 1)
    pub X: Arr2<f64>, // (N*T, K)
    pub T: usize,
    pub N: usize,
    pub K: usize,
    pub beta: Arr1<f64>,
    pub beta_cov: Arr2<f64>,
    pub u: Arr2<f64>
}


impl PlrBase {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr1<f64>, X: Arr2<f64>, T: usize ) -> Result<Self, &'static str> {
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
        let beta = Arr1::<f64>::new(K);
        let beta_cov = Arr2::<f64>::new(K, K);
        let u = Arr2::<f64>::new(N, T);
        Ok( Self{ y, X, T, N, K, beta, beta_cov, u } )
    }

    #[inline]
    pub fn get_u( &mut self ) {
        self.u.copy(&self.y);
        dgemv!(-1., &self.X, &self.beta, 1., &mut self.u);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn t_test( &self, k: usize, H0: f64 ) -> (f64, f64) {
        let df = (self.N*self.T - self.K) as f64;
        let distr = statrs::distribution::StudentsT::new(0., 1., df).unwrap();
        let test_stat = (self.beta[k] - H0) / self.beta_cov[(k, k)].sqrt();
        let pvalue = 2.* distr.cdf( -test_stat.abs() );
        (test_stat, pvalue)
    }

    #[inline] #[allow(non_snake_case)]
    pub fn z_test( &self, k: usize, H0: f64 ) -> (f64, f64) {
        let distr = statrs::distribution::Normal::new(0., 1.).unwrap();
        let test_stat = (self.beta[k] - H0) / self.beta_cov[(k, k)].sqrt();
        let pvalue = 2.* distr.cdf( -test_stat.abs() );
        (test_stat, pvalue)
    }

    #[inline]
    pub fn reset( &mut self ) {
        self.beta.reset();
        self.beta_cov.reset();
    }

}


pub struct PlrOls {
    pub base: PlrBase,
    pub sigma2: f64
}

impl PlrOls {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr1<f64>, X: Arr2<f64>, T: usize ) -> Result<Self, &'static str> {
        let base: PlrBase = PlrBase::new(y, X, T)?;
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
        let mut V = Arr2::<f64>::new(self.base.T, self.base.T);
        self.get_V(&mut V);
        self.get_beta_cov_sandwich(&mut V);
    }

    #[inline]
    pub fn get_beta( &mut self ) {
        dsyrk!(1., &self.base.X, 0., &mut self.base.beta_cov, trans); // Get beta_icov = X^T*X
        dgemv!(1., &self.base.X, &self.base.y, 0., &mut self.base.beta, trans); // Get beta <- X^T * y
        dposv!(&mut self.base.beta_cov, &mut self.base.beta); // Get beta_icov_lo and solve beta.
    }

    #[inline]
    pub fn get_sigma2( &mut self ) {
        self.sigma2 = self.base.u.sumsquare() / (self.base.N * self.base.T - self.base.K) as f64; 
    }

    #[inline]
    pub fn get_beta_cov( &mut self ) {
        dpotri!(&mut self.base.beta_cov); // Get beta_cov.
        self.base.beta_cov.scale(self.sigma2);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_V( &self, V: &mut Arr2<f64> ) {
        dsyrk!(1./self.base.N as f64, &self.base.u, 0., V, trans);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_beta_cov_sandwich( &mut self, V0: &mut Arr2<f64> ) {
        dpotrf!(V0);
   
        let mut Xi = Arr2::<f64>::new(self.base.T, self.base.K);
        let A = self.base.beta_cov.clone();
        self.base.beta_cov.reset();
        let mut AB = Arr2::<f64>::new(self.base.K, self.base.K);
        let mut sel: Vec<usize> = vec![0; self.base.T];

        for i in 0..self.base.N {
            for (sel_, t) in zip(sel.as_mut_slice(), 0..self.base.T) {
                *sel_ = i + t * self.base.N;
            }
            Xi.get_rows(&self.base.X, sel.as_slice());
            dtrmm!(1./self.sigma2, V0, &mut Xi, left, trans);
            dsyrk!(1., &Xi, 1., &mut self.base.beta_cov, trans);
        }
        self.base.beta_cov.copy_lower_to_upper();
        dsymm!(1., &A, &self.base.beta_cov, 0., &mut AB, left);
        dsymm!(1., &A, &AB, 0., &mut self.base.beta_cov, right);
    }

}


#[allow(non_snake_case)]
pub struct PlrGls {
    pub base: PlrBase,
    pub V: Arr2<f64>,
    pub V_lo: Arr2<f64>,
    pub V_inv: Arr2<f64>,
    pub V_invlo: Arr2<f64>
}

impl PlrGls {

    #[inline] #[allow(non_snake_case)]
    pub fn new( y: Arr1<f64>, X: Arr2<f64>, T: usize, V: Arr2<f64> ) -> Result<Self, &'static str> {
        let base: PlrBase = PlrBase::new(y, X, T)?;
        let mut V_lo = V.clone();
        dpotrf!(&mut V_lo);
        let mut V_inv = V_lo.clone();
        dpotri!(&mut V_inv);
        let mut V_invlo = V_inv.clone();
        dpotrf!(&mut V_invlo);
        Ok( Self{ base, V, V_lo, V_inv, V_invlo } )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn new_feasible( y: Arr1<f64>, X: Arr2<f64>, T: usize ) -> Result<Self, &'static str> {
        let mut m: PlrOls = PlrOls::new(y, X, T)?;
        m.get_beta();
        m.base.get_u();
        let mut V = Arr2::<f64>::new(m.base.T, m.base.T);
        m.get_V(&mut V);
        Ok( PlrGls::new( m.base.y, m.base.X, T, V )? )
    }

    #[inline] #[allow(non_snake_case)]
    pub fn new_uninit( y: Arr1<f64>, X: Arr2<f64>, T: usize ) -> Result<Self, &'static str> {
        let base: PlrBase = PlrBase::new(y, X, T)?;
        let V = Arr2::<f64>::new(T, T);
        let V_lo = Arr2::<f64>::new(T, T);
        let V_inv = Arr2::<f64>::new(T, T);
        let V_invlo = Arr2::<f64>::new(T, T);
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
        dpotrf!(&mut self.V_lo);

        self.V_inv.copy(&self.V_lo);
        dpotri!(&mut self.V_inv);

        self.V_invlo.copy(&self.V_inv);
        dpotrf!(&mut self.V_invlo);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_beta( &mut self ) {
        // Compute beta_icov and X^T * iV * y.
        let mut yi = Arr1::<f64>::new(self.base.T);
        let mut Xi = Arr2::<f64>::new(self.base.T, self.base.K);
        let mut V_inv_yi = Arr1::<f64>::new(self.base.T);
        let mut sel: Vec<usize> = vec![0; self.base.T];

        for i in 0..self.base.N {
            for (sel_, t) in zip(sel.as_mut_slice(), 0..self.base.T) {
                *sel_ = i + t * self.base.N;
            }
            yi.get_elements(&self.base.y, sel.as_slice());
            Xi.get_rows(&self.base.X, sel.as_slice());

            dsymv!(1., &self.V_inv, &yi, 0., &mut V_inv_yi);
            dgemv!(1., &Xi, &V_inv_yi, 1., &mut self.base.beta, trans);

            dtrmm!(1., &self.V_invlo, &mut Xi, left, trans);
            dsyrk!(1., &Xi, 1., &mut self.base.beta_cov, trans); // It is actually beta_icov now.
        }
        // Compute beta.
        dposv!(&mut self.base.beta_cov, &mut self.base.beta); // It is actually beta_icov_L now. And beta is solved.
    }

    #[inline]
    pub fn get_beta_cov( &mut self ) {
        dpotri!(&mut self.base.beta_cov); // Get beta_cov.
    }

}
