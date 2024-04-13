use fan455_arrf64::*;
use std::marker::PhantomData;


#[inline]
pub fn logaddexp( x: f64, y: f64 ) -> f64 {
    let tmp: f64 = x - y;
    if tmp > 0. {
        x + (-tmp).exp().ln_1p()
    } else {
        y + tmp.exp().ln_1p()
    }
}


#[inline]
pub fn logaddexp_update( x: &mut f64, y: f64 ) {
    let tmp: f64 = *x - y;
    if tmp > 0. {
        *x += (-tmp).exp().ln_1p();
    } else {
        *x = y + tmp.exp().ln_1p();
    }
}


pub struct CubicPoly {
    p0: f64, p1: f64, p2: f64, p3: f64, tol: f64, max_iter: usize
    // p0 * x^3 + p1 * x^2 + p2 * x + p3 = 0
}


impl CubicPoly
{
    #[inline]
    pub fn new( p0: f64, p1: f64, p2: f64, p3: f64 ) -> Self {
        Self { p0, p1, p2, p3, tol: 1e-6, max_iter: 1000 }
    }

    #[inline]
    pub fn with_tol( mut self, val: f64 ) -> Self { self.tol = val; self }

    #[inline]
    pub fn with_max_iter( mut self, val: usize ) -> Self { self.max_iter = val; self }

    #[inline]
    pub fn find( &self, mut x0: f64 ) -> Option<f64> {
        let mut x1: f64; let mut fx: f64; let mut dfx: f64;
        let mut total_iter: usize = self.max_iter;
        for i in 0..self.max_iter {
            fx = self.p0 * x0.powi(3) + self.p1 * x0.powi(2) + self.p2 * x0 + self.p3;
            dfx = 3.*self.p0 + x0.powi(2) + 2.*self.p1 * x0 + self.p2;
            x1 = x0 - fx / dfx;
            if (x1 - x0).abs() < self.tol {
                total_iter = i; 
                break;
            }
            x0 = x1;
        }
        match total_iter != self.max_iter {
            true => Some(x0),
            false => None,
        }
    }

    pub fn find_with_range( &self, x0: f64, lb: f64, ub: f64 ) -> Option<f64> {
        match self.find(x0) {
            Some(x) => match x >= lb && x <= ub {
                true => Some(x),
                false => None,
            }
            None => None,
        }
    }
}


pub trait MvNewtonFunc<VT1: RVec<f64>, VT2: RVecMut<f64>, MT: RMatMut<f64>>
{
    fn f_df( &self, x: &VT1, fx: &mut VT2, dfx: &mut MT );
    fn dim( &self ) -> usize;
}

pub struct MvNewton<'a, F, VT1, VT2, MT>
where 
    F: MvNewtonFunc<VT1, VT2, MT>,
    VT1: RVecMut<f64>,
    VT2: RVecMut<f64> + RMatMut<f64> + GVecAlloc<f64>,
    MT: RMatMut<f64> + GMatAlloc<f64>,
{
    tol: f64, 
    max_iter: usize,
    fun: &'a F,
    fx: VT2,
    dfx: MT,
    ph1: PhantomData<VT1>,
}


impl<'a, F, VT1, VT2, MT> MvNewton<'a, F, VT1, VT2, MT>
where 
    F: MvNewtonFunc<VT1, VT2, MT>,
    VT1: RVecMut<f64>,
    VT2: RVecMut<f64> + RMatMut<f64> + GVecAlloc<f64>,
    MT: RMatMut<f64> + GMatAlloc<f64>,
{
    #[inline]
    pub fn new( fun: &'a F ) -> Self {
        let dim = fun.dim();
        Self {
            tol: 1e-6, 
            max_iter: 1000,
            fun,
            fx: VT2::alloc(dim),
            dfx: MT::alloc(dim, dim),
            ph1: PhantomData::<VT1>,
        }
    }

    #[inline]
    pub fn solve( &mut self, x: &mut VT1 ) -> Option<()> {
        let mut err: f64;
        let mut total_iter: usize = self.max_iter;

        for i in 0..self.max_iter {
            self.fun.f_df(&x, &mut self.fx, &mut self.dfx);
            err = self.fx.norm();
            if err < self.tol {
                total_iter = i; 
                break;
            }
            dgesv(&mut self.dfx, &mut self.fx);
            x.subassign(&self.fx);
        }
        match total_iter != self.max_iter {
            true => Some(()),
            false => None,
        }
    }
}