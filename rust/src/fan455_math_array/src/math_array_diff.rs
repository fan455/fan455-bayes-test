use fan455_math_scalar::*;
use fan455_arrf64::*;
use fan455_util::{elem, mzip, assert_multi_eq};


#[derive(Clone)]
pub struct NumGrad { // Central differentiation
    pub dim: usize,
    pub x: Vec<f64>,
    pub h: Vec<f64>,
}

impl NumGrad
{
    #[inline]
    pub fn default_h() -> f64 {
        f64::EPSILON.sqrt()
    }

    #[inline]
    pub fn new( dim: usize ) -> Self {
        let x: Vec<f64> = vec![0.; dim];
        let h: Vec<f64> = vec![Self::default_h(); dim];
        Self { dim, x, h }
    }

    #[inline]
    pub fn call_oneside(
        &mut self, f: fn(&[f64])->f64, x0: &[f64], y0: f64, grad: &mut [f64],
    ) {
        self.x.copy_sl(x0);
        let mut y1: f64;
        let mut h: f64;
        //assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.len());
        for k in 0..self.dim {
            h = self.h[k];
            self.x[k] += h; // x+h
            y1 = f(&self.x); // f(x+h)
            grad[k] = (y1 - y0) / h;
            self.x[k] -= h; // x
        }
    }

    #[inline]
    pub fn call_twoside(
        &mut self, f: fn(&[f64])->f64, x0: &[f64], _y0: f64, grad: &mut [f64],
    ) {
        self.x.copy_sl(x0);
        let mut y1: f64;
        let mut y2: f64;
        let mut h: f64;
        //assert_multi_eq!(self.dim, self.x.size(), self.h.size(), grad.len());
        for k in 0..self.dim {
            h = self.h[k];
            self.x[k] += h; // x+h
            y1 = f(&self.x); // f(x+h)
            self.x[k] -= 2.* h; // x-h
            y2 = f(&self.x); // f(x-h)
            grad[k] = (y1 - y2) / (2.* h);
            self.x[k] += h; // x
        }
    }
}
