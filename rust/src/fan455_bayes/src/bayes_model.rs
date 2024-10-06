use fan455_arrf64::*;


#[allow(non_snake_case)]
pub struct PsPoint {
    // Phase point
    pub q: Arr<f64>,
    pub p: Arr<f64>,
    pub g: Arr<f64>,
    pub E: f64,
}

impl PsPoint {

    #[inline]
    pub fn new( n: usize ) -> Self {
        Self {
            q: Arr::<f64>::new(n),
            p: Arr::<f64>::new(n),
            g: Arr::<f64>::new(n),
            E: 0_f64
        }
    }

    #[inline]
    pub fn new_copy( z: &Self ) -> Self {
        Self {
            q: Arr::<f64>::new_copy(&z.q),
            p: Arr::<f64>::new_copy(&z.p),
            g: Arr::<f64>::new_copy(&z.g),
            E: z.E
        }
    }

    #[inline]
    pub fn copy( &mut self, z: &Self ) {
        self.q.copy(&z.q);
        self.p.copy(&z.p);
        self.g.copy(&z.g);
        self.E = z.E
    }
}


pub trait BayesModel
{
    fn get_dim( &self ) -> usize;
    
    fn init_hmc(
        &mut self,
        z_q: &mut Arr<f64>,
        inv_metric: &mut Mat<f64>, 
        metric: &mut Mat<f64>, 
        metric_lo: &mut Mat<f64> 
    );

    fn update(
        &mut self,
        z: &mut PsPoint
    );
    // Compute potential energy, gradient and Hessian.
}
