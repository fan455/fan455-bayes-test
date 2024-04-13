/* 
Implentation of Stan NUTS algorithm in Rust.
Reference: https://github.com/stan-dev/stan
*/

use fan455_arrf64::*;
use fan455_arrf64_macro::*;
use rand::{Rng, RngCore};
use rand::distributions::Standard;
use rand_distr::{Distribution, StandardNormal};
use super::bayes_model_base::{PsPoint, BayesModel};
use super::bayes_math::{logaddexp, logaddexp_update};


#[allow(non_snake_case)]
pub fn nuts_dense<Model: BayesModel, RngType: Rng>(
    rng: RngType,
    model: Model,
    max_depth: usize,
    max_deltaH: f64,
    delta: f64,
    gamma: f64,
    kappa: f64,
    t0: f64,
    epsilon_init: f64,
    n_warmup: usize, 
    init_buffer: usize, 
    base_window: usize, 
    term_buffer: usize,
    n_prog_adapt: usize, 
    n_prog_draw: usize,
    sample_buffer: &mut Arr2<f64>,
) {
    let mut sampler = NutsDense::new(
        rng, model, max_depth, max_deltaH, delta, gamma, kappa, t0, epsilon_init
    );
    let mut tmp = NutsBuf::new(sampler.dim);
    sampler.init();
    sampler.adapt(&mut tmp, n_warmup, n_prog_adapt, init_buffer, base_window, term_buffer);
    sampler.draw(&mut tmp, sample_buffer, n_prog_draw);
}

pub struct NutsBuf {
    z_fwd: PsPoint,
    z_bck: PsPoint,
        
    z_sample: PsPoint,
    z_propose: PsPoint,
        
    p_fwd_fwd: Arr1<f64>,
    p_sharp_fwd_fwd: Arr1<f64>,
        
    p_fwd_bck: Arr1<f64>,
    p_sharp_fwd_bck: Arr1<f64>,
        
    p_bck_fwd: Arr1<f64>,
    p_sharp_bck_fwd: Arr1<f64>,
        
    p_bck_bck: Arr1<f64>,
    p_sharp_bck_bck: Arr1<f64>,
    
    rho: Arr1<f64>,
    rho_fwd: Arr1<f64>,
    rho_bck: Arr1<f64>,
    rho_extended: Arr1<f64>,
    rho_subtree: Arr1<f64>,
}


impl NutsBuf {

    #[inline]
    pub fn new(n: usize) -> Self {
      Self {
        z_fwd: PsPoint::new(n),
        z_bck: PsPoint::new(n),
            
        z_sample: PsPoint::new(n),
        z_propose: PsPoint::new(n),
            
        p_fwd_fwd: Arr1::<f64>::new(n),
        p_sharp_fwd_fwd: Arr1::<f64>::new(n),
            
        p_fwd_bck: Arr1::<f64>::new(n),
        p_sharp_fwd_bck: Arr1::<f64>::new(n),
            
        p_bck_fwd: Arr1::<f64>::new(n),
        p_sharp_bck_fwd: Arr1::<f64>::new(n),
            
        p_bck_bck: Arr1::<f64>::new(n),
        p_sharp_bck_bck: Arr1::<f64>::new(n),
        
        rho: Arr1::<f64>::new(n),
        rho_fwd: Arr1::<f64>::new(n),
        rho_bck: Arr1::<f64>::new(n),
        rho_extended: Arr1::<f64>::new(n),
        rho_subtree: Arr1::<f64>::new(n),
      }  
    }

}


#[allow(non_snake_case)]
pub struct NutsDense<Model: BayesModel, RngType: Rng> {
    pub model: Model,

    dim: usize,
    z: PsPoint,
    inv_e_metric: Arr2<f64>,
    e_metric: Arr2<f64>,
    e_metric_lo: Arr2<f64>,

    max_depth: usize,
    max_deltaH: f64,
    depth: usize,
    n_leapfrog: usize,
    accept_prob: f64,
    divergent: bool,
    
    delta: f64,
    gamma: f64,
    kappa: f64,
    t0: f64,
    epsilon_init: f64,
    epsilon: f64,
    mu: f64,
    s_bar: f64,
    logepsilon_bar: f64,

    n_sample: usize,
    n_warmup: usize,
    init_buffer: usize,
    term_buffer: usize,
    base_window: usize,
    total_window: usize,
    n_window: usize,
    win_sizes: Vec<usize>,

    step_count: f64,
    win_count: f64,
    delta1: Arr1<f64>,
    delta2: Arr1<f64>,
    m1: Arr1<f64>,
    m2: Arr2<f64>,

    pub rng: RngType,
    distr_stduf: Standard,
    distr_stdnorm: StandardNormal,
}


impl<Model: BayesModel, RngType: RngCore> NutsDense<Model, RngType> {

    #[inline] #[allow(non_snake_case)]
    pub fn new(
        rng: RngType,
        model: Model,
        max_depth: usize,
        max_deltaH: f64,
        delta: f64,
        gamma: f64,
        kappa: f64,
        t0: f64,
        epsilon_init: f64
    ) -> Self {
        let dim: usize = model.get_dim();
        Self {
            model,
            dim,
            z: PsPoint::new(dim),
            inv_e_metric: Arr2::<f64>::new(dim, dim),
            e_metric: Arr2::<f64>::new(dim, dim),
            e_metric_lo: Arr2::<f64>::new(dim, dim),
        
            max_depth,
            max_deltaH,
            depth: 0,
            n_leapfrog: 0,
            accept_prob: 0.,
            divergent: false,
            
            delta,
            gamma,
            kappa,
            t0,
            epsilon_init,
            epsilon: epsilon_init,
            mu: 0.,
            s_bar: 0.,
            logepsilon_bar: 0.,
        
            n_sample: 0,
            n_warmup: 0,
            init_buffer: 0,
            term_buffer: 0,
            base_window: 0,
            total_window: 0,
            n_window: 0,
            win_sizes: Vec::new(),
        
            step_count: 0.,
            win_count: 0.,
            delta1: Arr1::<f64>::new(dim),
            delta2: Arr1::<f64>::new(dim),
            m1: Arr1::<f64>::new(dim),
            m2: Arr2::<f64>::new(dim, dim),
        
            rng,
            distr_stduf: Standard,
            distr_stdnorm: StandardNormal,
        }
    }
    

    #[inline]
    pub fn init( &mut self ) {
        self.model.init_hmc(
            &mut self.z, 
            &mut self.inv_e_metric, 
            &mut self.e_metric, 
            &mut self.e_metric_lo
        );
        self.model.f_df(&mut self.z);
        #[cfg(feature="nuts-enable-msg")] println!("Initialization completed.");
    }


    #[inline]
    pub fn reset( &mut self ) {
        self.epsilon = self.epsilon_init;
        self.restart_stepsize();
        self.restart_window();
    }


    #[inline] #[allow(unused_variables)]
    pub fn adapt(
        &mut self,
        tmp: &mut NutsBuf,
        n_warmup: usize, 
        n_prog: usize, 
        init_buffer: usize, 
        base_window: usize, 
        term_buffer: usize
    ) {
        // Calaulate window.
        self.n_warmup = n_warmup;
        self.init_buffer = init_buffer;
        self.term_buffer = term_buffer;
        self.base_window = base_window;
        self.total_window = self.n_warmup - self.init_buffer - self.term_buffer;
        let before_term: usize = self.init_buffer + self.total_window;
    
        // Compute the number of windows.
        self.n_window = (1. + self.total_window as f64 / 
            self.base_window as f64).log2().floor() as usize;
    
        // Compute the sizes of windows.
        let mut win_check_arr: Vec<usize> = Vec::with_capacity(self.n_window+1);
        self.win_sizes.reserve(self.n_window);
        let mut win_size: usize = self.base_window;
        let mut win_check: usize = self.init_buffer + self.base_window - 1;
    
        for _i in 0..self.n_window-1 {
            self.win_sizes.push(win_size);
            win_check_arr.push(win_check);
            win_size *= 2;
            win_check += win_size;
        }
        self.win_sizes.push(win_size + before_term - win_check - 1);
        win_check_arr.push(self.init_buffer + self.total_window - 1);
        win_check_arr.push(0);
    
        // Calculate progress.
        #[cfg(feature="nuts-enable-msg")] let mut prog_check_arr: Vec<usize> = Vec::with_capacity(n_prog + 1);
        #[cfg(feature="nuts-enable-msg")] let prog_size: usize = self.n_warmup / n_prog;
        #[cfg(feature="nuts-enable-msg")] let mut prog_check: usize = prog_size - 1;
        #[cfg(feature="nuts-enable-msg")]
        for _i in 0..n_prog-1 {
            prog_check_arr.push(prog_check);
            prog_check += prog_size;
        }
        #[cfg(feature="nuts-enable-msg")] {prog_check_arr.push(self.n_warmup - 1);}
        #[cfg(feature="nuts-enable-msg")] {prog_check_arr.push(0);}
        
        // Begin adaptation.
        let mut i_win: usize = 0;
        win_check = win_check_arr[i_win];
    
        #[cfg(feature="nuts-enable-msg")] let mut i_prog: usize = 0;
        #[cfg(feature="nuts-enable-msg")] {prog_check = prog_check_arr[i_prog];}

        // Initial buffer.
        self.init_stepsize();
    
        for i in 0..self.init_buffer {
            self.transition(tmp);
            self.learn_stepsize();
    
            #[cfg(feature="nuts-enable-msg")]
            if i == prog_check {
                i_prog += 1;
                prog_check = prog_check_arr[i_prog];
                println!("Adaptation: {} / {}", i_prog, n_prog);
            }
        }
        
        // Window buffer.
        for i in self.init_buffer..before_term {
            self.transition(tmp);
            self.learn_stepsize();
            self.prepare_covar();
    
            if i == win_check {
                self.learn_covar();
                self.restart_stepsize();
                self.restart_window();
                self.init_stepsize();
        
                i_win += 1;
                win_check = win_check_arr[i_win];
            }
    
            #[cfg(feature="nuts-enable-msg")]
            if i == prog_check {
                i_prog += 1;
                prog_check = prog_check_arr[i_prog];
                println!("Adaptation: {} / {}", i_prog, n_prog);
            }
        }
    
        // Terminal buffer.
        for i in before_term..self.n_warmup {
            self.transition(tmp);
            self.learn_stepsize();
    
            #[cfg(feature="nuts-enable-msg")]
            if i == prog_check {
                i_prog += 1;
                prog_check = prog_check_arr[i_prog];
                println!("Adaptation: {} / {}", i_prog, n_prog);
            }
        }
    
        // End adaption.
        self.epsilon = self.logepsilon_bar.exp();
    }


    #[inline] #[allow(unused_variables)]
    pub fn draw( 
        &mut self,
        tmp: &mut NutsBuf,
        buffer: &mut Arr2<f64>, 
        n_prog: usize 
    ) {
         // buffer: (dim, n_sample), samples of z.q
        self.n_sample = buffer.ncol();
    
        // Calculate progress.
        #[cfg(feature="nuts-enable-msg")] let mut prog_check_arr: Vec<usize> = Vec::with_capacity(n_prog+1);
        #[cfg(feature="nuts-enable-msg")] let prog_size: usize = self.n_sample / n_prog;
        #[cfg(feature="nuts-enable-msg")] let mut prog_check: usize = prog_size - 1;
        #[cfg(feature="nuts-enable-msg")]
        for _i in 0..n_prog-1 {
            prog_check_arr.push(prog_check);
            prog_check += prog_size;
        }
        #[cfg(feature="nuts-enable-msg")] {prog_check_arr.push(self.n_sample-1);}
        #[cfg(feature="nuts-enable-msg")] {prog_check_arr.push(0);}

        // Begin sampling.
        #[cfg(feature="nuts-enable-msg")] let mut i_prog: usize = 0;
        #[cfg(feature="nuts-enable-msg")] {prog_check = prog_check_arr[i_prog];}
        
        for i in 0..self.n_sample {
            self.transition(tmp);
            buffer.col_mut(i).copy(&self.z.q);
    
            #[cfg(feature="nuts-enable-msg")]
            if i == prog_check {
                i_prog += 1;
                prog_check = prog_check_arr[i_prog];
                println!("Sampling: {} / {}", i_prog, n_prog)
            }
        }
    }


    #[inline]
    fn prepare_covar( &mut self ) {
        self.win_count += 1.;

        self.delta1.assign_sub(&self.z.q, &self.m1);
        daxpy(1./self.win_count, &self.delta1, &mut self.m1);

        self.delta2.assign_sub(&self.z.q, &self.m1);
        dger(1., &self.delta2, &self.delta1, &mut self.m2);
    }


    #[inline]
    fn learn_covar( &mut self ) {

        #[cfg(feature="nuts_covar_adjust_num_of_samples")]
        self.inv_e_metric.assign_scale(
            &self.m2, self.win_count/((self.win_count+5.)*(self.win_count-1.)),
        );

        #[cfg(not(feature="nuts_covar_adjust_num_of_samples"))]
        self.inv_e_metric.assign_scale(&self.m2, 1./(self.win_count-1.));
        
        #[cfg(feature="nuts_covar_regularization")]
        self.inv_e_metric.addassign_iden(0.001*5./(self.win_count+5.));
        
        self.e_metric.copy(&self.inv_e_metric);
        dpotrf!(&mut self.e_metric);
        dpotri!(&mut self.e_metric);
    
        self.e_metric_lo.copy(&self.e_metric);
        dpotrf!(&mut self.e_metric_lo);
    }


    #[inline] #[allow(non_snake_case)]
    fn init_stepsize( &mut self ) {
        // Skip initialization for extreme step sizes
        if self.epsilon == 0. || self.epsilon > 1.0e7 || self.epsilon.is_nan() {return;}

        let z_init = PsPoint::new_copy(&self.z);
        self.sample_p();
        let mut H0: f64 = self.hamiltonian();
        self.leapfrog(self.epsilon);
        let mut h: f64 = self.hamiltonian();
        if h.is_nan() { h = f64::INFINITY; }

        let mut delta_H: f64 = H0 - h;
        let direction: i32 = match delta_H > 0.8_f64.ln() {
            true => 1,
            false => -1,
        };

        loop {
            self.z.copy(&z_init);
            self.sample_p();
            H0 = self.hamiltonian();
            self.leapfrog(self.epsilon);
            h = self.hamiltonian();
            if h.is_nan() { h = f64::INFINITY; }

            delta_H = H0 - h;
            if (direction == 1) && (delta_H < 0.8_f64.ln()) {break;}
            if (direction == -1) && (delta_H > 0.8_f64.ln()) {break;}
            self.epsilon = match direction == 1 {
                true => 2.* self.epsilon,
                false => 0.5* self.epsilon,
            };

            if self.epsilon > 1.0e7 {
                panic!("Posterior is improper. Please check your model.");
            } 
            if self.epsilon == 0. {
                panic!("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
            }
        }
        self.z.copy(&z_init);
        self.mu = (10.* self.epsilon).ln();
    }


    #[inline]
    fn learn_stepsize( &mut self ) {
        self.step_count += 1.;
    
        let mut tmp: f64 = 1. / (self.step_count + self.t0);
        self.s_bar = (1. - tmp) * self.s_bar + tmp * (self.delta - self.accept_prob);
        let logepsilon: f64 = self.mu - self.s_bar * self.step_count.sqrt() / self.gamma;
              
        tmp = self.step_count.powf(-self.kappa);
        self.logepsilon_bar = (1. - tmp) * self.logepsilon_bar + tmp * logepsilon;
    
        self.epsilon = logepsilon.exp();
    }


    #[inline]
    fn restart_stepsize( &mut self ) {
        self.step_count = 0.;
        self.s_bar = 0.;
        self.logepsilon_bar = 0.;
    }
    

    #[inline]
    fn restart_window( &mut self ) {
        self.win_count = 0.;
        self.m1.reset();
        self.m2.reset();
    }


    #[inline]
    fn sample_uf( &mut self ) -> f64 {
        self.distr_stduf.sample(&mut self.rng)
    }


    #[inline]
    fn sample_p( &mut self ) {
        for p_ in self.z.p.itm() {
            *p_ = self.distr_stdnorm.sample(&mut self.rng);
        }
        dtrmv!(&self.e_metric_lo, &mut self.z.p);
    }


    #[inline]
    fn leapfrog( &mut self, epsilon: f64 ) {
        daxpy(-0.5*epsilon, &self.z.g, &mut self.z.p);
        dsymv!(epsilon, &self.inv_e_metric, &self.z.p, 1., &mut self.z.q); 
        self.model.f_df(&mut self.z);
        daxpy(-0.5*epsilon, &self.z.g, &mut self.z.p);
    }


    #[inline]
    fn dtau_dp( &self, p_sharp: &mut Arr1<f64> ) {
        dsymv!(1., &self.inv_e_metric, &self.z.p, 0., p_sharp);
    }


    #[inline]
    fn hamiltonian( &self ) -> f64 {
        let mut p_sharp = Arr1::<f64>::new(self.dim);
        dsymv!(1., &self.inv_e_metric, &self.z.p, 0., &mut p_sharp);
        0.5 * ddot(&self.z.p, &p_sharp) + self.z.V
    }


    #[inline]
    fn hamiltonian2( &self, p_sharp: &Arr1<f64> ) -> f64 {
        0.5 * ddot(&self.z.p, p_sharp) + self.z.V
    }


    #[inline]
    fn compute_criterion(
        &self,
        p_sharp_minus: &Arr1<f64>,
        p_sharp_plus: &Arr1<f64>,
        rho: &Arr1<f64>
    ) -> bool {
        ddot(p_sharp_plus, rho) > 0. && ddot(p_sharp_minus, rho) > 0.
    }


    #[inline] #[allow(non_snake_case)]
    fn build_tree(
        &mut self,
        depth: usize, 
        z_propose: &mut PsPoint, 
        p_sharp_beg: &mut Arr1<f64>, 
        p_sharp_end: &mut Arr1<f64>, 
        rho: &mut Arr1<f64>, 
        rho_subtree: &mut Arr1<f64>, 
        p_beg: &mut Arr1<f64>, 
        p_end: &mut Arr1<f64>, 
        H0: f64, 
        sign: f64, 
        n_leapfrog: &mut usize, 
        log_sum_weight: &mut f64, 
        sum_metro_prob: &mut f64
    ) -> bool {
        // Base case
        if depth == 0 {
            self.leapfrog(sign*self.epsilon);
            *n_leapfrog += 1;
        
            self.dtau_dp(p_sharp_beg);
            p_sharp_end.copy(p_sharp_beg);
        
            rho.addassign(&self.z.p);
            p_beg.copy(&self.z.p);
            p_end.copy(p_beg);
        
            let delta_H: f64 = H0 - self.hamiltonian2(p_sharp_beg);
            if delta_H + self.max_deltaH < 0. {
                self.divergent = true;
            }
            logaddexp_update(log_sum_weight, delta_H);
            if delta_H > 0. { *sum_metro_prob += 1.; }
            else { *sum_metro_prob += delta_H.exp(); }
        
            z_propose.copy(&self.z);

            return !self.divergent;
        }
        // General recursion
        // Build the initial subtree
        let mut log_sum_weight_init: f64 = f64::NEG_INFINITY;
    
        // Momentum and sharp momentum at end of the initial subtree
        let mut p_init_end = Arr1::<f64>::new(self.dim);
        let mut p_sharp_init_end = Arr1::<f64>::new(self.dim);
        let mut rho_init = Arr1::<f64>::new(self.dim);
    
        let valid_init: bool = self.build_tree(
            depth-1, 
            z_propose, 
            p_sharp_beg, 
            &mut p_sharp_init_end, 
            &mut rho_init, 
            rho_subtree, 
            p_beg, 
            &mut p_init_end, 
            H0, 
            sign, 
            n_leapfrog, 
            &mut log_sum_weight_init, 
            sum_metro_prob
        );
        if valid_init == false {return false;}
    
        // Build the final subtree
        let mut z_propose_final = PsPoint::new_copy(&self.z);
        let mut log_sum_weight_final: f64 = f64::NEG_INFINITY;
    
        // Momentum and sharp momentum at beginning of the final subtree
        let mut p_final_beg = Arr1::<f64>::new(self.dim);
        let mut p_sharp_final_beg = Arr1::<f64>::new(self.dim);
        let mut rho_final = Arr1::<f64>::new(self.dim);
            
        let valid_final: bool = self.build_tree(
            depth-1, 
            &mut z_propose_final, 
            &mut p_sharp_final_beg, 
            p_sharp_end, 
            &mut rho_final, 
            rho_subtree, 
            &mut p_final_beg, 
            p_end, 
            H0, 
            sign, 
            n_leapfrog, 
            &mut log_sum_weight_final, 
            sum_metro_prob
        );
        if valid_final == false {return false;}
    
        // Multinomial sample from right subtree
        let log_sum_weight_subtree: f64 = logaddexp(log_sum_weight_init, log_sum_weight_final);
        logaddexp_update(log_sum_weight, log_sum_weight_subtree);
    
        if log_sum_weight_final > log_sum_weight_subtree {
            z_propose.copy(&z_propose_final);
        } else if self.sample_uf() < (log_sum_weight_final - log_sum_weight_subtree).exp() {
            z_propose.copy(&z_propose_final);
        }
        rho_subtree.assign_add(&rho_init, &rho_final);
        rho.addassign(rho_subtree);
    
        // Demand satisfaction around merged subtrees
        let mut persist_criterion: bool = self.compute_criterion(p_sharp_beg, p_sharp_end, &rho_subtree);
    
        // Demand satisfaction between subtrees
        rho_subtree.assign_add(&rho_init, &p_final_beg);
        persist_criterion &= self.compute_criterion(p_sharp_beg, &p_sharp_final_beg, &rho_subtree);
    
        rho_subtree.assign_add(&rho_final, &p_init_end);
        persist_criterion &= self.compute_criterion(&p_sharp_init_end, p_sharp_end, &rho_subtree);
    
        persist_criterion
      }


    #[inline] #[allow(non_snake_case)]
    fn transition( &mut self, tmp: &mut NutsBuf ) {
        // Initialize the algorithm
        self.sample_p();
    
        tmp.z_fwd.copy(&self.z);
        tmp.z_bck.copy(&self.z);
    
        tmp.z_sample.copy(&self.z);
        tmp.z_propose.copy(&self.z);
    
        tmp.p_fwd_fwd.copy(&self.z.p);
        self.dtau_dp(&mut tmp.p_sharp_fwd_fwd);
    
        tmp.p_fwd_bck.copy(&self.z.p);
        tmp.p_sharp_fwd_bck.copy(&tmp.p_sharp_fwd_fwd);
    
        tmp.p_bck_fwd.copy(&self.z.p);
        tmp.p_sharp_bck_fwd.copy(&tmp.p_sharp_fwd_fwd);
    
        tmp.p_bck_bck.copy(&self.z.p);
        tmp.p_sharp_bck_bck.copy(&tmp.p_sharp_fwd_fwd);
    
        // Integrated momenta along trajectory
        tmp.rho.copy(&self.z.p);

        let mut log_sum_weight: f64 = 0.;
        let H0: f64 = self.hamiltonian2(&tmp.p_sharp_fwd_fwd);
        let mut n_leapfrog: usize = 0;
        let mut sum_metro_prob: f64 = 0.;
    
        self.depth = 0;
        self.divergent = false;
    
        while self.depth < self.max_depth {
            // Build a new subtree in a random direction
            tmp.rho_fwd.reset();
            tmp.rho_bck.reset();
        
            let valid_subtree: bool;
            let mut log_sum_weight_subtree: f64 = f64::NEG_INFINITY;
        
            if self.sample_uf() > 0.5 {
                // Extend the current trajectory forward
                self.z.copy(&tmp.z_fwd);
                tmp.rho_bck.copy(&tmp.rho);
                tmp.p_bck_fwd.copy(&tmp.p_fwd_fwd);
                tmp.p_sharp_bck_fwd.copy(&tmp.p_sharp_fwd_fwd);
        
                valid_subtree = self.build_tree(
                    self.depth, 
                    &mut tmp.z_propose, 
                    &mut tmp.p_sharp_fwd_bck, 
                    &mut tmp.p_sharp_fwd_fwd, 
                    &mut tmp.rho_fwd,
                    &mut tmp.rho_subtree,  
                    &mut tmp.p_fwd_bck, 
                    &mut tmp.p_fwd_fwd, 
                    H0, 
                    1., 
                    &mut n_leapfrog, 
                    &mut log_sum_weight_subtree, 
                    &mut sum_metro_prob
                );
                tmp.z_fwd.copy(&self.z);

            } else {
                // Extend the current trajectory backwards
                self.z.copy(&tmp.z_bck);
                tmp.rho_fwd.copy(&tmp.rho);
                tmp.p_fwd_bck.copy(&tmp.p_bck_bck);
                tmp.p_sharp_fwd_bck.copy(&tmp.p_sharp_bck_bck);
        
                valid_subtree = self.build_tree(
                    self.depth, 
                    &mut tmp.z_propose, 
                    &mut tmp.p_sharp_bck_fwd, 
                    &mut tmp.p_sharp_bck_bck, 
                    &mut tmp.rho_bck, 
                    &mut tmp.rho_subtree, 
                    &mut tmp.p_bck_fwd, 
                    &mut tmp.p_bck_bck, 
                    H0, 
                    -1., 
                    &mut n_leapfrog, 
                    &mut log_sum_weight_subtree, 
                    &mut sum_metro_prob
                );
                tmp.z_bck.copy(&self.z);
            }
            if valid_subtree == false {break;}
        
            // Sample from accepted subtree
            self.depth += 1;
        
            if log_sum_weight_subtree > log_sum_weight {
                tmp.z_sample.copy(&tmp.z_propose);
            } else if self.sample_uf() < (log_sum_weight_subtree - log_sum_weight).exp() {
                tmp.z_sample.copy(&tmp.z_propose);
            }
            logaddexp_update(&mut log_sum_weight, log_sum_weight_subtree);
        
            // Break when no-u-turn criterion is no longer satisfied
            tmp.rho.assign_add(&tmp.rho_bck, &tmp.rho_fwd);
        
            // Demand satisfaction around merged subtrees
            let mut persist_criterion: bool = self.compute_criterion(&tmp.p_sharp_bck_bck, &tmp.p_sharp_fwd_fwd, &tmp.rho);
        
            // Demand satisfaction between subtrees
            tmp.rho_extended.assign_add(&tmp.rho_bck, &tmp.p_fwd_bck);
            persist_criterion &= self.compute_criterion(&tmp.p_sharp_bck_bck, &tmp.p_sharp_fwd_bck, &tmp.rho_extended);
        
            tmp.rho_extended.assign_add(&tmp.rho_fwd, &tmp.p_bck_fwd);
            persist_criterion &= self.compute_criterion(&tmp.p_sharp_bck_fwd, &tmp.p_sharp_fwd_fwd, &tmp.rho_extended);

            if persist_criterion == false {break;}
        }
        self.n_leapfrog = n_leapfrog;
        self.accept_prob = sum_metro_prob / n_leapfrog as f64;
        self.z.copy(&tmp.z_sample);
      }


}