use fan455_arrf64::*;


#[allow(non_snake_case)]
pub struct PsPoint {
    pub q: Arr1<f64>,
    pub p: Arr1<f64>,
    pub g: Arr1<f64>,
    pub V: f64,
}


impl PsPoint {

    #[inline]
    pub fn new( n: usize ) -> Self {
        Self {
            q: Arr1::<f64>::new(n),
            p: Arr1::<f64>::new(n),
            g: Arr1::<f64>::new(n),
            V: 0_f64
        }
    }


    #[inline]
    pub fn new_copy( z: &Self ) -> Self {
        Self {
            q: Arr1::<f64>::new_copy(&z.q),
            p: Arr1::<f64>::new_copy(&z.p),
            g: Arr1::<f64>::new_copy(&z.g),
            V: z.V
        }
    }

    #[inline]
    pub fn copy( &mut self, z: &Self ) {
        self.q.copy(&z.q);
        self.p.copy(&z.p);
        self.g.copy(&z.g);
        self.V = z.V
    }

}


pub trait BayesModel {

    fn get_dim( &self ) -> usize;
    

    fn init_hmc(
        &mut self,
        z: &mut PsPoint, 
        inv_e_metric: &mut Arr2<f64>, 
        e_metric: &mut Arr2<f64>, 
        e_metric_lo: &mut Arr2<f64> 
    );

    fn f_df(
        &mut self,
        z: &mut PsPoint
    );

}