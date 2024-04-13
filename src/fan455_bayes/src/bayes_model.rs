use fan455_arrf64::*;
use fan455_arrf64_macro::*;
use super::bayes_model_base::{PsPoint, BayesModel};
use super::bayes_stat::ols;


#[allow(non_snake_case)]
pub struct PlrBayes
{
    pub y: Arr1<f64>, // (N*T, 1)
    pub X: Arr2<f64>, // (N*T, K)
    pub T: usize,
    pub N: usize,
    pub K: usize,
    pub Nf: f64,
    pub u: Arr2<f64>, // (N*T, 1), can be (N, T)
    pub u2: Arr2<f64>, // (N, T), can be (N*T, 1)
    pub W: Arr2<f64>, // (T, T)
    pub beta_ols: Arr1<f64>, // (K, K)
    pub beta_cov_ols: Arr2<f64>, // (K, K)
    pub beta_icov_ols: Arr2<f64>, // (K, K)
    pub u_cov_ols: Arr2<f64>, // (T, T)
    pub u_icov_ols: Arr2<f64>, // (T, T)
}


impl PlrBayes
{
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
        Ok( Self {
            y, X, T, N, K, 
            Nf: N as f64,
            u: Arr2::<f64>::new(N, T),
            u2: Arr2::<f64>::new(N, T),
            W: Arr2::<f64>::new(T, T),
            beta_ols: Arr1::<f64>::new(K),
            beta_cov_ols: Arr2::<f64>::new(K, K),
            beta_icov_ols: Arr2::<f64>::new(K, K),
            u_cov_ols: Arr2::<f64>::new(T, T),
            u_icov_ols: Arr2::<f64>::new(T, T)
        } )
    }

}


impl BayesModel for PlrBayes
{
    #[inline]
    fn get_dim( &self ) -> usize {
        self.K
    }
    

    #[inline] #[allow(non_snake_case)]
    fn init_hmc(
        &mut self,
        z: &mut PsPoint, 
        inv_e_metric: &mut Arr2<f64>, 
        e_metric: &mut Arr2<f64>, 
        e_metric_lo: &mut Arr2<f64> 
    ) {
        ols(&self.y, &self.X, &mut self.beta_ols);
        z.q.copy(&self.beta_ols);

        self.u.copy(&self.y);
        dgemv!(-1., &self.X, &self.beta_ols, 1., &mut self.u);

        dsyrk!(1./self.Nf, &self.u, 0., &mut self.u_cov_ols, trans);

        self.u_icov_ols.copy(&self.u_cov_ols);
        dpotrf!(&mut self.u_icov_ols);
        dpotri!(&mut self.u_icov_ols);

        let mut u_icov_ols_lo = Arr2::<f64>::new_copy(&self.u_icov_ols);
        dpotrf!(&mut u_icov_ols_lo);
    
        let mut Xi = Arr2::<f64>::new(self.T, self.K);
        let mut rows: Vec<usize> = vec![0; self.T];

        for i in 0..self.N {
            let mut t: usize = 0;
            for r_ in rows.iter_mut() {
                *r_ = i + t * self.N;
                t += 1;
            }
            Xi.get_rows(&self.X, &rows);
            dtrmm!(1., &u_icov_ols_lo, &mut Xi, trans);
            dsyrk!(1., &Xi, 1., &mut self.beta_icov_ols, trans);
        }

        e_metric.copy(&self.beta_icov_ols);

        self.beta_cov_ols.copy(&self.beta_icov_ols);
        dpotrf!(&mut self.beta_cov_ols);
        e_metric_lo.copy(&self.beta_cov_ols);
    
        dpotri!(&mut self.beta_cov_ols);
        inv_e_metric.copy(&self.beta_cov_ols);
    }


    #[inline]
    fn f_df(
        &mut self,
        z: &mut PsPoint
    ) {
        self.u.copy(&self.y);
        dgemv!(-1., &self.X, &z.q, 1., &mut self.u);
        dsyrk!(1./self.Nf, &self.u, 0., &mut self.W, trans);
        dpotrf!(&mut self.W);

        z.V = self.Nf * self.W.sumlogdiag();

        dpotri!(&mut self.W);
        dsymm!(1., &self.W, &self.u, 0., &mut self.u2, right);
        dgemv!(-1., &self.X, &self.u2, 0., &mut z.g, trans);
    }
}
