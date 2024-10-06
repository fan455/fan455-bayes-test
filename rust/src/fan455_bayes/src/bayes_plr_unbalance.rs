use std::usize;
use fan455_arrf64::*;
use fan455_util::*;
use super::bayes_model::{PsPoint, BayesModel};
use super::bayes_stat::*;


#[allow(non_snake_case)]
pub struct BayesUnbalanceGroup {
    pub idx: usize,
    pub n_group: usize,
    pub unit_range: [usize; 2],
    pub pat: Vec<bool>, // (T,)
    pub time_idx: Vec<usize>, // (T,), global time index.
    pub cover_groups: Vec<usize>,
    pub covered_by_groups: Vec<usize>,

    pub T_global: usize,
    pub T: usize,
    pub N_global: usize,
    pub N: usize,
    pub N_: f64,
    pub NT: usize,
    pub K: usize,
    pub N0: usize,
    pub N0_: f64,

    pub X_aug: Mat<f64>, // (N*T_global, K)
    pub X: Mat<f64>, // (N*T, K)
    pub u_aug: Mat<f64>, // (N, T_global)
    pub u_aug_: Mat<f64>, // (N, T_global)
    pub u: Mat<f64>, // (N, T)
    pub u_: Mat<f64>, // (N, T)

    pub W_aug: Mat<f64>, // (T_global, T_global)
    pub W: Mat<f64>, // (T, T)
    pub Q: Mat<f64>, // (T_global, T_global)
    pub B: Mat<f64>, // (T_global, T)
    pub B_: Mat<f64>, // (T_global, T)
    pub Z: Mat<f64>, // (T, T)
    pub Z_: Mat<f64>, // (T, T), buffer
    pub Z__: Mat<f64>, // (T, T), buffer
}

impl BayesUnbalanceGroup
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        i_group: usize,
        patterns: &Mat<bool>,
        unit_range_global: &[[usize; 2]],
        X_global: &Mat<f64>,
        N_global: usize,
        K: usize,
    ) -> Self {
        let idx = i_group;
        let unit_range = unit_range_global[i_group];
        let N = unit_range[1] - unit_range[0];
        
        let T_global = patterns.nrow();
        let n_group = patterns.ncol();
        let pat = patterns.col(i_group).to_vec();
        let T = pat.iter().filter(|&x| *x).count();
        let mut time_idx = Vec::<usize>::with_capacity(T);
        for (t, pat_) in pat.iter().enumerate() {
            if *pat_ {
                time_idx.push(t);
            }
        }
        assert_eq!(time_idx.len(), T);
        let N_ = N as f64;
        let NT = N * T;

        //println!("patterns = {:?}", &patterns.data);
        let mut cover_groups = Vec::<usize>::with_capacity(n_group);
        let mut covered_by_groups = Vec::<usize>::with_capacity(n_group);
        for g in 0..n_group {
            let mut cover_: bool = true;
            'loop_1: for elem!(this, other) in mzip!(pat.it(), patterns.col(g).it()) {
                if !this && *other {
                    cover_ = false;
                    break 'loop_1;
                }
            }
            if cover_ {
                cover_groups.push(g);
            }
            let mut covered_: bool = true;
            'loop_2: for elem!(this, other) in mzip!(pat.it(), patterns.col(g).it()) {
                if *this && !other {
                    covered_ = false;
                    break 'loop_2;
                }
            }
            if covered_ {
                covered_by_groups.push(g);
            }
        }
        cover_groups.shrink_to_fit();
        covered_by_groups.shrink_to_fit();
        //println!("i_group = {i_group}, cover_groups = {cover_groups:?}");

        let mut N0: usize = 0;
        for g in covered_by_groups.iter() {
            let range_ = unit_range_global[*g];
            N0 += range_[1] - range_[0];
        }
        let N0_ = N0 as f64;

        let Q = Mat::<f64>::new(T_global, T_global);
        let W_aug = Mat::<f64>::new(T_global, T_global);
        let W = Mat::<f64>::new(T, T);
        let Z = Mat::<f64>::new(T, T);
        let Z_ = Mat::<f64>::new(T, T);
        let Z__ = Mat::<f64>::new(T, T);

        let mut X_aug = Mat::<f64>::new(N*T_global, K);
        {
            let mut it = X_aug.itm();
            for k in 0..K {
                let Xk = X_global.col_as_mat(k, N_global, T_global);
                for t in 0..T_global {
                    let Xk_ = Xk.col(t);
                    for Xk__ in Xk_.subvec(unit_range[0], unit_range[1]).it() {
                        *it.next().unwrap() = *Xk__;
                    }
                }
            }
        }
        let mut X = Mat::<f64>::new(NT, K);
        {
            let mut it = X.itm();
            for k in 0..K {
                let Xk = X_global.col_as_mat(k, N_global, T_global);
                for t in time_idx.iter() {
                    let Xk_ = Xk.col(*t);
                    for Xk__ in Xk_.subvec(unit_range[0], unit_range[1]).it() {
                        *it.next().unwrap() = *Xk__;
                    }
                }
            }
        }
        let u_aug = Mat::<f64>::new(N, T_global);
        let u_aug_ = u_aug.clone();
        let u = Mat::<f64>::new(N, T);
        let u_ = u.clone();

        let mut B = Mat::<f64>::new(T_global, T);
        for (t, t_global) in time_idx.iter().enumerate() {
            *B.idxm2(*t_global, t) = 1.;
        }
        let B_ = B.clone();

        Self { 
            idx, n_group, cover_groups, covered_by_groups, unit_range, pat, time_idx, 
            T_global, T, N_global, N, N_, N0, N0_, NT, K, 
            X_aug, X, u_aug, u_aug_, u, u_, W_aug, W, Q, B, B_, Z, Z_, Z__, 
        }
    }

    #[inline]
    pub fn get_u( &mut self, u_global: &Mat<f64> ) {
        for (t, t_global) in self.time_idx.iter().enumerate() {
            self.u.col_mut(t).copy(&u_global.subvec2(
                self.unit_range[0], *t_global, self.unit_range[1], *t_global)
            );
        }
    }

    #[inline]
    pub fn get_u_aug( &mut self, u_global: &Mat<f64> ) {
        for t in 0..self.T_global {
            self.u_aug.col_mut(t).copy(&u_global.subvec2(
                self.unit_range[0], t, self.unit_range[1], t)
            );
        }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn get_W( &mut self ) {
        dsymm(1., &self.W_aug, &self.B, 0., &mut self.B_, LEFT, LOWER);
        dgemmt(1., &self.B, &self.B_, 0., &mut self.W, TRANS, NO_TRANS, LOWER);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn update_W_aug( &mut self, W_buf: &Mat<f64> ) {
        daxpy(1./self.N0_, W_buf, &mut self.W_aug);
    }
}


#[allow(non_snake_case)]
pub struct BayesUnbalanceGroupBuf
{
    pub Q_buf: Mat<f64>, // (T, T)
    pub N0_: f64,
    pub N_: f64,
    pub cover_groups: Vec<usize>,
}


#[allow(non_snake_case)]
pub struct BayesPlrUnbalance
{
    pub n_group: usize,
    pub groups: Vec<BayesUnbalanceGroup>, // (n_group,)
    pub groups_buf: Vec<BayesUnbalanceGroupBuf>, // (n_group,)
    pub unit_range: Vec<[usize; 2]>, // (n_group,)

    pub dim: usize,
    pub T: usize,
    pub N: usize,
    pub N_: f64,
    pub N_vec: Vec<usize>, // (T,)
    pub NT: usize,
    pub K: usize,
    pub n_obs: usize,

    pub y: Arr<f64>, // (N*T, 1), ordered by N
    pub X: Mat<f64>, // (N*T, K), ordered by N
    pub u: Mat<f64>, // (N, T), ordered by N
    pub u_T: Mat<f64>, // (T, N), ordered by T
    pub W_buf: Mat<f64>, // (T, T)

    pub n_update: usize,
}

impl BayesPlrUnbalance
{
    #[inline] #[allow(non_snake_case)]
    pub fn new( 
        y_aug_path: &str, 
        X_aug_path: &str,
        pat_path: &str,
        num_path: &str, // How many units each group has.
    ) -> Self {
        // Read data.
        let y = Arr::<f64>::read_npy(y_aug_path);
        let X = Mat::<f64>::read_npy(X_aug_path);
        let K = X.ncol();
        let NT = y.size();
        assert_eq!(NT, X.nrow());

        let mut pat = Mat::<bool>::read_npy(pat_path);
        pat.ensure_col_vec();
        let T = pat.nrow();
        let n_group = pat.ncol();
        assert_eq!(NT % T, 0);
        let N = NT / T;
        let N_ = N as f64;

        let group_N = read_npy_vec::<usize>(num_path);
        assert_eq!(group_N.len(), n_group);
        let mut unit_range = Vec::<[usize; 2]>::with_capacity(n_group);
        {
            let mut beg: usize = 0;
            let mut end: usize = 0;
            for N_g in group_N.iter() {
                end += N_g;
                unit_range.push([beg, end]);
                beg = end;
            }
            assert_eq!(unit_range.len(), n_group);
        }

        let mut n_obs: usize = 0;
        let mut N_vec = vec![0_usize; T];
        let mut groups = Vec::<BayesUnbalanceGroup>::with_capacity(n_group);
        for i in 0..n_group {
            let group = BayesUnbalanceGroup::new(
                i, &pat, &unit_range, &X, N, K
            );
            n_obs += group.NT;
            for t in group.time_idx.iter() {
                N_vec[*t] += group.N;
            }
            groups.push(group);
        }
        let mut groups_buf = Vec::<BayesUnbalanceGroupBuf>::with_capacity(n_group);
        for group in groups.iter() {
            groups_buf.push(BayesUnbalanceGroupBuf {
                Q_buf: Mat::<f64>::new(T, T), N_: group.N_, N0_: group.N0_, 
                cover_groups: group.cover_groups.clone(),
            });
        }

        let dim = K;
        let n_update: usize = 0;

        let u = Mat::<f64>::new(N, T);
        let u_T = Mat::<f64>::new(T, N);
        let W_buf = Mat::<f64>::new(T, T);

        println!("N = {N}, T = {T}, K = {K}");
        println!("n_group = {n_group}, n_obs = {n_obs}");
        println!("N_vec = {N_vec:?}");

        Self { 
            n_group, groups, groups_buf, unit_range, 
            dim, T, N, N_, N_vec, NT, K, n_obs, 
            y, X, u, u_T, W_buf, n_update,
        }
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_u<VT: RVec<f64>>( &mut self, beta: &VT ) {
        self.u.copy(&self.y);
        dgemv(-1., &self.X, beta, 1., &mut self.u, NO_TRANS);
    }

    #[inline] #[allow(non_snake_case)]
    pub fn compute_local_W_aug( &mut self ) {
        // Compute local W_aug.
        for group in self.groups.iter_mut() {
            group.W_aug.reset();
        }

        for elem!(range, this) in mzip!(self.unit_range.iter(), self.groups_buf.iter()) {
            for i in range[0]..range[1] {
                self.W_buf.reset();
                dsyr(1., &self.u_T.col(i), &mut self.W_buf, LOWER);
                
                for g in this.cover_groups.iter() {
                    self.groups[*g].update_W_aug(&self.W_buf);
                }
            }
        }
    }

    #[inline]
    pub fn write_data( &self, folder: &str ) {
        let mut cover_groups = Mat::<bool>::new(self.n_group, self.n_group);
        let mut covered_by_groups = Mat::<bool>::new(self.n_group, self.n_group);
        let mut groups_units_base = Vec::<usize>::with_capacity(self.n_group);
        let mut groups_units = Vec::<usize>::with_capacity(self.n_group);

        for (g, group) in self.groups.iter().enumerate() {
            groups_units_base.push(group.N0);
            groups_units.push(group.N);

            for g_ in group.cover_groups.iter() {
                *cover_groups.idxm2(*g_, g) = true;
            }
            for g_ in group.covered_by_groups.iter() {
                *covered_by_groups.idxm2(*g_, g) = true;
            }
        }
        write_npy_vec(&format!("{folder}/units_each_year.npy"), &self.N_vec);
        cover_groups.write_npy(&format!("{folder}/cover_groups.npy"));
        covered_by_groups.write_npy(&format!("{folder}/covered_by_groups.npy"));
        write_npy_vec(&format!("{folder}/base_units_each_group.npy"), &groups_units_base);
        write_npy_vec(&format!("{folder}/units_each_group.npy"), &groups_units);
    }
}


impl BayesModel for BayesPlrUnbalance
{
    #[inline]
    fn get_dim( &self ) -> usize {
        self.dim
    }
    
    #[inline] #[allow(non_snake_case)]
    fn init_hmc(
        &mut self,
        z_q: &mut Arr<f64>,
        inv_metric: &mut Mat<f64>, 
        metric: &mut Mat<f64>, 
        metric_lo: &mut Mat<f64> 
    ) {
        metric.reset();
        ols(&self.y, &self.X, z_q);

        // Compute global u.
        self.compute_u(z_q);

        // Transpose u to u_T
        self.u_T.get_trans(&self.u);

        // Compute local W_aug.
        self.compute_local_W_aug();

        // Compute potential energy and gradient.
        for group in self.groups.iter_mut() {
            // Compute local W.
            group.get_W();

            // Factorize local W
            let dpotrf_info = dpotrf(&mut group.W, LOWER);
            if dpotrf_info != 0 {
                let pr = RMatPrinter::default();
                let pr1 = RVecPrinter::default();
                println!("dpotrf_info = {}", dpotrf_info);
                pr1.print("z.q", z_q);
                pr.print("W_L", &group.W);
                panic!(
                    "Error: model.init_hmc(), W is not positive definite, n_update = {}, i_group = {}", 
                    self.n_update, group.idx
                );
            }
            dpotri(&mut group.W, LOWER);
            dpotrf(&mut group.W, LOWER);

            // Compute local u.
            group.get_u(&self.u);

            let mut Xi = Mat::<f64>::new(group.T, self.K);
            let mut rows: Vec<usize> = vec![0; group.T];
    
            for i in 0..group.N {
                let mut t: usize = 0;
                for r_ in rows.iter_mut() {
                    *r_ = i + t * group.N;
                    t += 1;
                }
                Xi.get_rows(&group.X, &rows);
                dtrmm(1., &group.W, &mut Xi, LEFT, TRANS, LOWER);
                dsyrk(1., &Xi, 1., metric, TRANS, LOWER);
            }
        }
        metric_lo.copy(metric);
        dpotrf(metric_lo, LOWER);

        inv_metric.copy(metric_lo);
        dpotri(inv_metric, LOWER);
    }


    #[inline] #[allow(non_snake_case)]
    fn update(
        &mut self,
        z: &mut PsPoint
    ) {
        // Reset
        z.E = 0.;
        z.g.reset();

        // Compute global u
        self.compute_u(&z.q);

        // Transpose u to u_T
        self.u_T.get_trans(&self.u);

        // Compute local W_aug.
        self.compute_local_W_aug();

        // Compute potential energy and gradient.
        for elem!(group, group_buf) in mzip!(self.groups.iter_mut(), self.groups_buf.iter_mut()) {
            let N_ = group.N_;

            // Compute local W.
            group.get_W();

            // Factorize local W
            let dpotrf_info = dpotrf(&mut group.W, LOWER);
            if dpotrf_info != 0 {
                let pr = RMatPrinter::default();
                let pr1 = RVecPrinter::default();
                println!("dpotrf_info = {}", dpotrf_info);
                pr1.print("z.q", &z.q);
                pr.print("W_L", &group.W);
                panic!(
                    "Error: model.update(), W is not positive definite, n_update = {}, i_group = {}", 
                    self.n_update, group.idx
                );
            }
            z.E += N_ * group.W.sumlogdiag();
            dpotri(&mut group.W, LOWER);

            // Compute local u.
            group.get_u(&self.u);

            // Compute local Z.
            dsyrk(1./N_, &group.u, 0., &mut group.Z, TRANS, LOWER);

            // Compute local Q_buf.
            group.Z.copy_lower_to_upper();
            dsymm(1., &group.W, &group.Z, 0., &mut group.Z_, RIGHT, LOWER);
            z.E += 0.5 * N_ * group.Z_.trace();
            group.Z_.copy_lower_to_upper();
            dsymm(1., &group.W, &group.Z_, 0., &mut group.Z__, LEFT, LOWER);
            daxpy(-1., &group.W, &mut group.Z__);
            dsymm(1., &group.Z__, &group.B, 0., &mut group.B_, RIGHT, LOWER);
            dgemmt(1., &group.B_, &group.B, 0., &mut group_buf.Q_buf, NO_TRANS, TRANS, LOWER);

            // Compute gradient, part 1..
            dsymm(1., &group.W, &group.u, 0., &mut group.u_, RIGHT, LOWER);
            dgemv(-1., &group.X, &group.u_, 1., &mut z.g, TRANS);
        }

        // Compute local Q.
        for group in self.groups.iter_mut() {
            group.Q.reset();

            for g in group.cover_groups.iter() {
                let other = &self.groups_buf[*g];
                daxpy(other.N_/other.N0_, &other.Q_buf, &mut group.Q);
            }

            // Compute gradient, part 2.
            // Compute local u_aug.
            group.get_u_aug(&self.u);
            dsymm(1., &group.Q, &group.u_aug, 0., &mut group.u_aug_, RIGHT, LOWER);
            dgemv(1., &group.X_aug, &group.u_aug_, 1., &mut z.g, TRANS);
        }

        // Record n_update.
        self.n_update += 1;
    }
}


impl BayesPlrUnbalance
// Frequentists methods.
{
    #[inline] #[allow(non_snake_case)]
    pub fn run_fgls( &mut self ) -> (Arr<f64>, Mat<f64>) {
        let mut beta = Arr::<f64>::new(self.K);
        let mut beta_cov = Mat::<f64>::new(self.K, self.K);

        ols(&self.y, &self.X, &mut beta);

        // Compute global u.
        self.compute_u(&beta);
        beta.reset();

        // Transpose u to u_T
        self.u_T.get_trans(&self.u);

        // Compute local W_aug.
        self.compute_local_W_aug();

        // Get a (N, T) matrix view of y.
        let y_mat = MatView::new(self.N, self.T, self.y.sl());

        // Compute potential energy and gradient.
        for group in self.groups.iter_mut() {
            // Compute local W.
            group.get_W();

            // Factorize local W
            assert!(dpotrf(&mut group.W, LOWER) == 0);
            dpotri(&mut group.W, LOWER);
            dpotrf(&mut group.W, LOWER);

            // Compute local y, and store it in u_ (attention!).
            for (t, t_global) in group.time_idx.iter().enumerate() {
                group.u_.col_mut(t).copy(&y_mat.subvec2(
                    group.unit_range[0], *t_global, group.unit_range[1], *t_global)
                );
            }

            let mut Xi = Mat::<f64>::new(group.T, self.K);
            let mut yi = Arr::<f64>::new(group.T);
            let mut sel: Vec<usize> = vec![0; group.T];
    
            for i in 0..group.N {
                let mut t: usize = 0;
                for r_ in sel.iter_mut() {
                    *r_ = i + t * group.N;
                    t += 1;
                }
                Xi.get_rows(&group.X, &sel);
                dtrmm(1., &group.W, &mut Xi, LEFT, TRANS, LOWER);
                dsyrk(1., &Xi, 1., &mut beta_cov, TRANS, LOWER);

                yi.get_elems(&group.u_, &sel);
                dtrmv(&group.W, &mut yi, TRANS, LOWER);
                dgemv(1., &Xi, &yi, 1., &mut beta, TRANS);
            }
        }
        dposv(&mut beta_cov, &mut beta, LOWER);
        dpotri(&mut beta_cov, LOWER);
        (beta, beta_cov)
    }
    

    #[inline] #[allow(non_snake_case)]
    pub fn run_ols( &mut self ) -> (Arr<f64>, Mat<f64>, Mat<f64>) {
        let mut beta = Arr::<f64>::new(self.K);
        let mut beta_cov = Mat::<f64>::new(self.K, self.K);
        let sigma2 = ols2(
            &self.y, &self.X, &mut self.u, &mut beta, &mut beta_cov, Some(self.n_obs)
        );

        let mut beta_cov_sw = Mat::<f64>::new(self.K, self.K);
        let mut buf_1 = Mat::<f64>::new(self.K, self.K);

        // Compute global u.
        self.compute_u(&beta);

        // Transpose u to u_T
        self.u_T.get_trans(&self.u);

        // Compute local W_aug.
        self.compute_local_W_aug();

        // Iterate groups.
        for group in self.groups.iter_mut() {
            let T = group.T;
            let N = group.N;

            // Compute local W.
            group.get_W();

            // Factorize local W
            assert!(dpotrf(&mut group.W, LOWER) == 0);
        
            let mut Xi = Mat::<f64>::new(T, self.K);
            let mut rows: Vec<usize> = vec![0; T];
    
            for i in 0..N {
                let mut t: usize = 0;
                for r_ in rows.iter_mut() {
                    *r_ = i + t * N;
                    t += 1;
                }
                Xi.get_rows(&group.X, &rows);
                dtrmm(1./sigma2, &group.W, &mut Xi, LEFT, TRANS, LOWER);
                dsyrk(1., &Xi, 1., &mut beta_cov_sw, TRANS, LOWER);
            }
        }
        beta_cov_sw.copy_lower_to_upper();
        dsymm(1., &beta_cov, &beta_cov_sw, 0., &mut buf_1, LEFT, LOWER);
        dsymm(1., &beta_cov, &buf_1, 0., &mut beta_cov_sw, RIGHT, LOWER);

        (beta, beta_cov, beta_cov_sw)
    }
}
