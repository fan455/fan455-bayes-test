use fan455_arrf64::*;
use super::bayes_model::{PsPoint, BayesModel};
use super::bayes_stat::ols;


#[allow(non_snake_case)]
pub struct BayesPlr
{
    pub y: Arr<f64>, // (N*T, 1), ordered by N
    pub X: Mat<f64>, // (N*T, K), ordered by N
    pub T: usize,
    pub N: usize,
    pub K: usize,
    pub Nf: f64,
    pub u: Mat<f64>, // (N*T, 1), can be (N, T)
    pub u2: Mat<f64>, // (N, T), can be (N*T, 1)
    pub W: Mat<f64>, // (T, T)
    pub beta_ols: Arr<f64>, // (K, K)
    pub beta_cov_ols: Mat<f64>, // (K, K)
    pub beta_icov_ols: Mat<f64>, // (K, K)
    pub u_cov_ols: Mat<f64>, // (T, T)
    pub u_icov_ols: Mat<f64>, // (T, T)
}

impl BayesPlr
{
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

        Ok( Self {
            y, X, T, N, K, 
            Nf: N as f64,
            u: Mat::<f64>::new(N, T),
            u2: Mat::<f64>::new(N, T),
            W: Mat::<f64>::new(T, T),
            beta_ols: Arr::<f64>::new(K),
            beta_cov_ols: Mat::<f64>::new(K, K),
            beta_icov_ols: Mat::<f64>::new(K, K),
            u_cov_ols: Mat::<f64>::new(T, T),
            u_icov_ols: Mat::<f64>::new(T, T)
        } )
    }
}


impl BayesModel for BayesPlr
{
    #[inline]
    fn get_dim( &self ) -> usize {
        self.K
    }
    

    #[inline] #[allow(non_snake_case)]
    fn init_hmc(
        &mut self,
        z_q: &mut Arr<f64>,
        inv_metric: &mut Mat<f64>, 
        metric: &mut Mat<f64>, 
        metric_lo: &mut Mat<f64> 
    ) {
        ols(&self.y, &self.X, &mut self.beta_ols);
        z_q.copy(&self.beta_ols);

        self.u.copy(&self.y);
        dgemv(-1., &self.X, &self.beta_ols, 1., &mut self.u, NO_TRANS);

        dsyrk(1./self.Nf, &self.u, 0., &mut self.u_cov_ols, TRANS, LOWER);

        self.u_icov_ols.copy(&self.u_cov_ols);
        dpotrf(&mut self.u_icov_ols, LOWER);
        dpotri(&mut self.u_icov_ols, LOWER);

        let mut u_icov_ols_lo = Mat::<f64>::new_copy(&self.u_icov_ols);
        dpotrf(&mut u_icov_ols_lo, LOWER);
    
        let mut Xi = Mat::<f64>::new(self.T, self.K);
        let mut rows: Vec<usize> = vec![0; self.T];

        for i in 0..self.N {
            let mut t: usize = 0;
            for r_ in rows.iter_mut() {
                *r_ = i + t * self.N;
                t += 1;
            }
            Xi.get_rows(&self.X, &rows);
            dtrmm(1., &u_icov_ols_lo, &mut Xi, LEFT, TRANS, LOWER);
            dsyrk(1., &Xi, 1., &mut self.beta_icov_ols, TRANS, LOWER);
        }

        metric.copy(&self.beta_icov_ols);

        self.beta_cov_ols.copy(&self.beta_icov_ols);
        dpotrf(&mut self.beta_cov_ols, LOWER);
        metric_lo.copy(&self.beta_cov_ols);
    
        dpotri(&mut self.beta_cov_ols, LOWER);
        inv_metric.copy(&self.beta_cov_ols);
    }


    #[inline] #[allow(non_snake_case)]
    fn update(
        &mut self,
        z: &mut PsPoint
    ) {
        self.u.copy(&self.y);
        dgemv(-1., &self.X, &z.q, 1., &mut self.u, NO_TRANS);
        dsyrk(1./self.Nf, &self.u, 0., &mut self.W, TRANS, LOWER);
        dpotrf(&mut self.W, LOWER);

        // compute potential
        z.E = self.Nf * self.W.sumlogdiag();

        // compute W^-1
        dpotri(&mut self.W, LOWER); 

        // compute gradient
        dsymm(1., &self.W, &self.u, 0., &mut self.u2, RIGHT, LOWER);
        dgemv(-1., &self.X, &self.u2, 0., &mut z.g, TRANS);
    }
}

