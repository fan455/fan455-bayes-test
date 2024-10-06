use fan455_arrf64::*;
use fan455_util::{elem, mzip};
use std::iter::zip;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, ChiSquared};
use super::bayes_math::{CubicPoly, MvNewtonFunc, MvNewton};
use statrs::distribution::ContinuousCDF;


#[inline] #[allow(non_snake_case)]
pub fn get_cor_ar1( T: usize, r: f64 ) -> Mat<f64> {
    let mut r_: f64 = 1.0;
    let mut cor = Mat::<f64>::new(T, T);
    cor.set_diag(r_);
    for t in 1..T {
        r_ *= r;
        cor.set_lband(t, r_);
        cor.set_uband(t, r_);
    }
    cor
}


#[inline]
pub fn cor_to_cov( cor: &Mat<f64>, scale: &Arr<f64> ) -> Mat<f64> {
    let mut cov = cor.clone();
    let n = cor.nrow();
    let mut diag = Mat::<f64>::new(n, n);
    diag.set_diag_to_vec(scale);
    dtrmm(1., &diag, &mut cov, LEFT, TRANS, LOWER);
    dtrmm(1., &diag, &mut cov, RIGHT, TRANS, LOWER);
    cov
}


#[inline]
pub fn cov_to_cor( cov: &Mat<f64> ) -> Mat<f64> {
    let mut cor = cov.clone();
    let n = cor.nrow();
    let mut x = cor.diag();
    for x_ in x.itm() {
        *x_ = x_.sqrt().recip();
    }
    let mut diag = Mat::<f64>::new(n, n);
    diag.set_diag_to_vec(&x);
    dtrmm(1., &diag, &mut cor, LEFT, TRANS, LOWER);
    dtrmm(1., &diag, &mut cor, RIGHT, TRANS, LOWER);
    cor
}


#[inline] #[allow(non_snake_case)]
pub fn infer_by_percent<VT: RVec<f64>>(
    sample: &VT, // Should have been sorted (lower to upper).
    H0: f64,
) -> (f64, f64) {
    let H0_percent = estimate_percent(sample, H0);
    let H0_pvalue = match H0_percent < 0.5 {
        true => 2.* H0_percent,
        false => 2.* (1. - H0_percent),
    };
    (H0_percent, H0_pvalue)
}


#[inline] #[allow(non_snake_case)]
pub fn infer_by_quantile<VT: RVec<f64>>(
    sample: &VT, // Should have been sorted (lower to upper).
    interval: f64,
) -> (f64, f64) {
    let alpha = (1.- interval) / 2.;
    let quantile_lower = estimate_quantile(sample, alpha, 0.4, 0.4);
    let quantile_upper = estimate_quantile(sample, alpha+interval, 0.4, 0.4);
    //let H0_reject: bool = H0 < quantile_lower || H0 > quantile_upper;
    (quantile_lower, quantile_upper)
}


#[inline] #[allow(non_snake_case)]
pub fn t_test_twotail( val: f64, H0: f64, df: f64, se: f64 ) -> (f64, f64) {
    let distr = statrs::distribution::StudentsT::new(0., 1., df).unwrap();
    let test_stat = (val - H0) / se;
    let pvalue = 2.* distr.cdf( -test_stat.abs() );
    (test_stat, pvalue)
}


#[inline] #[allow(non_snake_case)]
pub fn z_test_twotail( val: f64, H0: f64, se: f64 ) -> (f64, f64) {
    let distr = statrs::distribution::Normal::new(0., 1.).unwrap();
    let test_stat = (val - H0) / se;
    let pvalue = 2.* distr.cdf( -test_stat.abs() );
    (test_stat, pvalue)
}


#[inline] #[allow(non_snake_case)]
pub fn t_test_ci( val: f64, percent: f64, df: f64, se: f64 ) -> (f64, f64) {
    let distr = statrs::distribution::StudentsT::new(0., 1., df).unwrap();
    let test_stat = distr.inverse_cdf(percent+(1.-percent)/2.);
    (val-test_stat*se, val+test_stat*se)
}


#[inline] #[allow(non_snake_case)]
pub fn z_test_ci( val: f64, percent: f64, se: f64 ) -> (f64, f64) {
    let distr = statrs::distribution::Normal::new(0., 1.).unwrap();
    let test_stat = distr.inverse_cdf(percent+(1.-percent)/2.);
    (val-test_stat*se, val+test_stat*se)
}


#[inline] #[allow(non_snake_case)]
pub fn ols<VT1: RVec<f64>, VT2: RMatMut<f64>, MT: RMat<f64>>( y: &VT1, X: &MT, beta: &mut VT2 ) {
    // y: (N, 1), X: (N, K), beta: (K, 1)
    let mut A = Mat::<f64>::new(X.ncol(), X.ncol()); // (K, K)
    dsyrk(1., X, 0., &mut A, TRANS, LOWER);
    dgemv(1., X, y, 0., beta, TRANS);
    let info = dposv(&mut A, beta, LOWER);
    if info > 0 {
        panic!("bayes_stat::ols dposv got not positive definite error, info = {info}");
    }
}


#[inline] #[allow(non_snake_case)]
pub fn ols2<VT1: RVec<f64>, VT2: RMatMut<f64>, VT3: RMatMut<f64>, MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    y: &VT1, X: &MT1, u: &mut VT2, beta: &mut VT3, beta_cov: &mut MT2, n_obs: Option<usize>,
) -> f64 {
    // y: (N, 1), X: (N, K), beta: (K, 1), beta_cov: (K, K)
    assert_eq!(y.size(), X.nrow());

    dsyrk(1., X, 0., beta_cov, TRANS, LOWER);
    dgemv(1., X, y, 0., beta, TRANS);
    let info = dposv(beta_cov, beta, LOWER);
    if info > 0 {
        panic!("bayes_stat::ols2 dposv got not positive definite error, info = {info}");
    }

    u.copy(y);
    dgemv(-1., X, beta, 1., u, NO_TRANS);
    let K = X.ncol();
    let N = match n_obs {
        Some(N_) => N_,
        None => X.nrow(),
    };
    let sigma2 = u.sumsquare() / (N - K) as f64;

    dpotri(beta_cov, LOWER);
    beta_cov.scale(sigma2);

    sigma2
}


#[inline]
pub fn welford<VT: RVec<f64>>( x: &VT ) -> (f64, f64) {
    // Compute the mean and variance of x using Welford's method.
    let mut count: f64 = 0.;
    let mut mean: f64 = 0.;
    let mut var: f64 = 0.;
    let mut delta: f64;
    let mut delta2: f64;

    for x_ in x.it() {
        count += 1.;
        delta = x_ - mean;
        mean += delta / count;
        delta2 = x_ - mean;
        var += delta * delta2;
    }
    var /= count - 1.;
    (mean, var)
}

#[inline]
pub fn mv_welford<MT: RMat<f64>>( x: &MT ) -> (Arr<f64>, Mat<f64>) { // x: (dim, n_sample)
    let n: usize = x.nrow();
    let mut mean = Arr::<f64>::new(n);
    let mut cov = Mat::<f64>::new(n, n);
    let mut delta = Arr::<f64>::new(n);
    let mut delta2 = Arr::<f64>::new(n);
    mv_welford_with_buf(x, &mut mean, &mut cov, &mut delta, &mut delta2);
    (mean, cov)
}

#[inline]
pub fn mv_welford_with_buf<VT1: RVecMut<f64>, VT2: RVecMut<f64>, VT3: RVecMut<f64>, MT1: RMat<f64>, MT2: RMatMut<f64>>(
    x: &MT1,
    mean: &mut VT1,
    cov: &mut MT2,
    delta: &mut VT2,
    delta2: &mut VT3,
) { // x: (dim, n_sample)
    // !!! Before calling this function, please ensure mean is initialized with zeros!!!
    let mut count: f64 = 0.;
    for j in 0..x.ncol() {
        count += 1.;
        delta.assign_sub(&x.col(j), mean);
        daxpy(1./count, delta, mean);
        delta2.assign_sub(&x.col(j), mean);
        dger(1., delta, delta2, cov);
    }
    cov.scale(1./(count-1.));
}


#[inline]
pub fn estimate_percent<VT: RVec<f64>>( x: &VT, x0: f64 ) -> f64 {
    // x: sorted array; x0: value to find percent for.
    let mut p: usize = 0;
    for x_ in x.it() {
        if *x_ > x0 {break;}
        p += 1;
    }
    p as f64 / x.size() as f64
}


#[inline]
pub fn estimate_quantile<VT: RVec<f64>>(x: &VT, p: f64, alphap: f64, betap: f64) -> f64  {
    // Refer to scipy.stats.mstats.mquantiles.
    // x: sorted data array; p: quantile. alphap = 0.4, betap = 0.4
    let n: f64 = x.size() as f64;
    let m: f64 = alphap + p * (1. - alphap - betap);
    let j: f64 = (n * p + m).floor();
    let g: f64 = n * p + m - j;
    (1. - g) * x.idx(j as usize - 1) + g * x.idx(j as usize)
}


#[allow(dead_code)]
pub struct MvNormal {
    pub dim: usize,
    pub cov: Mat<f64>,
    pub cov_lo: Mat<f64>,
    pub distr_stdnorm: StandardNormal,
}

impl MvNormal
{
    #[inline]
    pub fn new( cov: Mat<f64> ) -> Self {
        let dim = cov.nrow();
        let mut cov_lo = cov.clone();
        dpotrf(&mut cov_lo, LOWER);
        Self { dim, cov, cov_lo, distr_stdnorm: StandardNormal }
    }

    #[inline]
    pub fn ndraw<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64> 
    ) {// x: (n_sample, dim)
        for x_ in x.itm() {
            *x_ = self.distr_stdnorm.sample(rng);
        }
        dtrmm(1., &self.cov_lo, x, RIGHT, TRANS, LOWER);
    }

    #[inline]
    pub fn ndraw_with_loc<RngType: Rng>(
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64>, 
        loc: &Arr<f64> 
    ) {// x: (n_sample, dim), loc: (dim,)
        self.ndraw(rng, x);
        x.addassign_rowvec(loc);
    }

    #[inline]
    pub fn ndraw_with_scale<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64>, 
        scale: &Arr<f64>
    ) {// x: (n_sample, dim), loc: (dim,)
        self.ndraw(rng, x);
        x.mulassign_rowvec(scale);
    }

    #[inline]
    pub fn ndraw_with_loc_scale<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64>, 
        loc: &Arr<f64>, 
        scale: &Arr<f64>
    ) {// x: (n_sample, dim), loc: (dim,)
        self.ndraw(rng, x);
        x.muladdassign_rowvec(scale, loc);
    }
}


#[allow(non_snake_case, dead_code)]
pub struct InvWishart // Refer to scipy's implementation.
{
    pub dim: usize,
    pub df: usize,
    pub S: Mat<f64>,
    pub S_lo: Mat<f64>,
    pub distr_stdnorm: StandardNormal,
    pub distr_chisq: Vec<ChiSquared<f64>>,
}

impl InvWishart {

    #[inline] #[allow(non_snake_case)]
    pub fn new( S: Mat<f64>, df: usize ) -> Self {
        let dim = S.nrow();
        let mut S_lo = S.clone();
        dpotrf(&mut S_lo, LOWER);
        S_lo.set_upper(0.);
        let distr_stdnorm = StandardNormal;
        let mut distr_chisq: Vec<ChiSquared<f64>> = Vec::with_capacity(dim);
        for df_chisq in df-dim+1..df+1 {
            distr_chisq.push(ChiSquared::new(df_chisq as f64).unwrap());
        }
        Self { dim, df, S, S_lo, distr_stdnorm, distr_chisq }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn new_with_mean( mut V: Mat<f64>, df: usize ) -> Self {
        V.scale((df - V.nrow() - 1) as f64);
        InvWishart::new(V, df)
    }

    #[inline] #[allow(non_snake_case)]
    pub fn draw<RngType: Rng>( 
        &self, 
        rng: &mut RngType 
    ) -> Mat<f64> {
        let mut W_inv: Mat<f64> = Mat::<f64>::new(self.dim, self.dim);
        let mut B = Mat::<f64>::new(self.dim, self.dim);
        self.draw_with_buf(rng, &mut W_inv, &mut B);
        W_inv
    }

    #[inline] #[allow(non_snake_case)]
    pub fn draw_with_buf<RngType: Rng>(
        &self, 
        rng: &mut RngType,
        W_inv: &mut Mat<f64>,
        B: &mut Mat<f64>,
    ) {
        B.clone_from(&self.S_lo);
        for (i, rv) in zip(0..self.dim, &self.distr_chisq) {
            *W_inv.idxm2(i,i) = rv.sample(rng).sqrt();
        }
        for j in 0..self.dim-1 {
            for i in j+1..self.dim {
                *W_inv.idxm2(i,j) = self.distr_stdnorm.sample(rng);
            }
        }
        dtrsm(1., W_inv, B, RIGHT, TRANS, LOWER); // Solve B.
        dsyrk(1., B, 0., W_inv, TRANS, LOWER); // W_inv = B * B^T
    }

}


pub struct NonNormalFunc {
    pub skew: f64,
    pub kurt: f64,
}

impl MvNewtonFunc<SArr1<f64, 3>, SArr1<f64, 3>, SArr2<f64, 3, 3, 9>> for NonNormalFunc
{
    #[inline]
    fn f_df( &self, x: &SArr1<f64, 3>, fx: &mut SArr1<f64, 3>, dfx: &mut SArr2<f64, 3, 3, 9> ) {
        let b = x[0];
        let c = x[1];
        let d = x[2];

        fx[0] = b.powi(2) + 6.* b * d + 2.* c.powi(2) + 15.* d.powi(2) - 1.;
        fx[1] = 2.* c * (b.powi(2) + 24.* b * d + 105.* d.powi(2) + 2.) - self.skew;
        fx[2] = 24. * (b * d + c.powi(2) * (1. + b.powi(2) + 28.* b * d) + d.powi(2) * (
            12. + 48.* b * d + 141.* c.powi(2) + 225.* d.powi(2)
        )) - self.kurt + 3.;
        
        dfx[(0, 0)] = 2.* b + 6.* d;
        dfx[(0, 1)] = 4.* c;
        dfx[(0, 2)] = 6.* b + 30.* d;
        dfx[(1, 0)] = 2.* c * (2.* b + 24.* d);
        dfx[(1, 1)] = 2.* (b.powi(2) + 24.* b * d + 105.* d.powi(2) + 2.);
        dfx[(1, 2)] = 2.* c * (24.* b + 210.* d);
        dfx[(2, 0)] = 24.* (d + c.powi(2) * (2.* b + 28.* d) + 48.* d.powi(3));
        dfx[(2, 1)] = 24.* (2.* c * (1. + b.powi(2) + 28.* b * d) + 282.* c * d.powi(2));
        dfx[(2, 2)] = 24.* (b + 28.* b * c.powi(2) + 24.* d + 144.* b * d.powi(2) + 
            282.* c.powi(2) * d + 900.* d.powi(3));
    }

    #[inline]
    fn dim( &self ) -> usize {
        3
    }
}


#[inline]
pub fn nonnormal_transform( z: f64, b: f64, c: f64, d: f64 ) -> f64 {
    -c + b * z + c * z.powi(2) + d * z.powi(2)
}


#[allow(dead_code)]
pub struct NonNormal {
    pub skew: f64,
    pub kurt: f64,
    pub b: f64, 
    pub c: f64, 
    pub d: f64,
    pub distr_stdnorm: StandardNormal,
}

impl NonNormal {

    #[inline]
    pub fn new( skew: f64, kurt: f64 ) -> Self {
        if kurt < skew.powi(2) + 1. {
            panic!("kurt >= skew^2 + 1 is required.");
        }
        let mut bcd: SArr1<f64, 3> = SArr1::from_array([1., 0., 0.]);
        let func = NonNormalFunc{skew, kurt};
        MvNewton::new(&func).solve(&mut bcd);
        Self { skew, kurt, b: bcd[0], c: bcd[1], d: bcd[2], distr_stdnorm: StandardNormal }
    }

    #[inline]
    pub fn ndraw_stdnorm<RngType: Rng>(
        &self,
        rng: &mut RngType,
        x: &mut Arr<f64>
    ) {
        for x_ in x.itm() {
            *x_ = self.distr_stdnorm.sample(rng);
        }
    }

    #[inline]
    pub fn ndraw<RngType: Rng>(
        &self,
        rng: &mut RngType,
        x: &mut Arr<f64>
    ) {
        self.ndraw_stdnorm(rng, x);
        let mut z: f64;
        for x_ in x.itm() {
            z = *x_;
            *x_ = nonnormal_transform(z, self.b, self.c, self.d); 
        }
    }

    #[inline]
    pub fn ndraw_with_loc_scale<RngType: Rng>(
        &self,
        rng: &mut RngType,
        x: &mut Arr<f64>,
        loc: f64,
        scale: f64
    ) {
        self.ndraw_stdnorm(rng, x);
        let mut z: f64;
        for x_ in x.itm() {
            z = *x_;
            *x_ = loc + scale * nonnormal_transform(z, self.b, self.c, self.d); 
        }
    }

}


#[allow(dead_code)]
pub struct MvNonNormal {
    pub dim: usize,
    pub skew: Arr<f64>,
    pub kurt: Arr<f64>,
    pub cor: Mat<f64>,
    pub b: Arr<f64>, 
    pub c: Arr<f64>, 
    pub d: Arr<f64>,
    pub rv_mvnorm: MvNormal,
}

impl MvNonNormal {

    #[inline]
    pub fn new( skew: Arr<f64>, kurt: Arr<f64>, cor: Mat<f64> ) -> Self {
        let dim: usize = skew.size();
        let mut b = Arr::<f64>::new(dim);
        let mut c = Arr::<f64>::new(dim);
        let mut d = Arr::<f64>::new(dim);
        for i in 0..dim {
            let rv = NonNormal::new(skew[i], kurt[i]);
            b[i] = rv.b; c[i] = rv.c; d[i] = rv.d;
        }
        let mut cor_mvnorm = Mat::<f64>::new(dim, dim);
        //cor_mvnorm.set_diag(1.);
        let mut p0: f64; let mut p1: f64; let mut p2 : f64; let mut p3: f64; 
        for j in 0..dim {
            for i in j..dim {        
                p0 = 6.* d[i] * d[j];
                p1 = 2.* c[i] * c[j];
                p2 = b[i] * b[j] + 3.* b[i] * d[j] + 3.* b[j] * d[i] + 9.* d[i] * d[j];
                p3 = -cor[(i,j)];
                //cor_mvnorm[(i,j)] = CubicPoly::new(p0, p1, p2, p3).find_with_range(0., -1., 1.).unwrap();
                cor_mvnorm[(i,j)] = CubicPoly::new(p0, p1, p2, p3).find(0.).unwrap();
            }
        }
        let rv_mvnorm = MvNormal::new(cor_mvnorm);
        Self { dim, skew, kurt, cor, b, c, d, rv_mvnorm }
    }


    #[inline]
    pub fn ndraw<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64> 
    ) {// x: (n_sample, dim)
        self.rv_mvnorm.ndraw(rng, x);
        let mut z: f64;
        for elem!(j, b_, c_, d_) in mzip!(0..x.ncol(), &self.b, &self.c, &self.d) {
            for x_ in x.col_mut(j) {
                z = *x_;
                *x_ = nonnormal_transform(z, *b_, *c_, *d_);
            }
        }
    }

    #[inline]
    pub fn ndraw_with_scale<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64>,
        scale: &Arr<f64>
    ) {// x: (n_sample, dim)
        self.rv_mvnorm.ndraw(rng, x);
        let mut z: f64;
        for elem!(j, b_, c_, d_, scale_) in mzip!(0..x.ncol(), &self.b, &self.c, &self.d, scale) {
            for x_ in x.col_mut(j) {
                z = *x_;
                *x_ = scale_ * nonnormal_transform(z, *b_, *c_, *d_);
            }
        }
    }

    #[inline]
    pub fn ndraw_with_loc_scale<RngType: Rng>( 
        &self, 
        rng: &mut RngType, 
        x: &mut Mat<f64>,
        loc: &Arr<f64>,
        scale: &Arr<f64>
    ) {// x: (n_sample, dim)
        self.rv_mvnorm.ndraw(rng, x);
        let mut z: f64;
        for elem!(j, b_, c_, d_, loc_, scale_) in mzip!(0..x.ncol(), &self.b, &self.c, &self.d, loc, scale) {
            for x_ in x.col_mut(j) {
                z = *x_;
                *x_ = loc_ + scale_ * nonnormal_transform(z, *b_, *c_, *d_);
            }
        }
    }

}