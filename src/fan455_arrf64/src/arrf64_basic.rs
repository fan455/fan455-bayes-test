use fan455_util::{elem, mzip};
use std::fmt::Display;
use std::iter::{zip, Iterator};
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};

#[derive(Clone, Copy)]
pub struct VecView<'a, T: General> {
    view: &'a [T],
}

pub struct VecViewMut<'a, T: General> {
    view: &'a mut [T],
}

#[derive(Clone, Copy)]
pub struct MatView<'a, T: General> {
    dim0: usize,
    dim1: usize,
    view: &'a [T],
}

pub struct MatViewMut<'a, T: General> {
    dim0: usize,
    dim1: usize,
    view: &'a mut [T],
}

pub struct DiagView<'a, T: General> {
    i: usize,
    dim: usize,
    view: &'a [T],
}

impl<'a, T: General> Iterator for DiagView<'a, T>
{
    type Item = &'a T;

    #[inline]
    fn next( &mut self ) -> Option<Self::Item> {
        if self.i < self.dim {
            self.i += 1;
            Some(self.view.index(self.i-1+(self.i-1)*self.dim))
        } else {
            None
        }
    }
}

pub trait GVecAlloc<T: General>
{
    fn alloc( n: usize ) -> Self;

    fn alloc_set( n: usize, val: T ) -> Self;

    fn alloc_copy<VT: GVec<T>>( x: &VT ) -> Self;
}

pub trait GMatAlloc<T: General>
{
    fn alloc( m: usize, n: usize ) -> Self;

    fn alloc_set( m: usize, n: usize, val: T ) -> Self;

    fn alloc_copy<MT: GMat<T>>( x: &MT ) -> Self;
}

pub trait GVec<T: General>
{
    fn size( &self ) -> usize;

    fn ptr( &self ) -> *const T;

    fn sl( &self ) -> &[T];

    fn idx( &self, i: usize ) -> &T;

    #[inline]
    fn it( &self ) -> Iter<T> {
        self.sl().iter()
    }
        
    #[inline]
    fn subvec( &self, i1: usize, i2: usize ) -> VecView<T> {
        VecView { view: &self.sl()[i1..i2] }
    }
}

pub trait GVecMut<T: General>: GVec<T>
{
    fn ptrm( &mut self ) -> *mut T;

    fn slm( &mut self ) -> &mut [T];

    fn idxm( &mut self, i: usize ) -> &mut T;

    #[inline]
    fn itm( &mut self ) -> IterMut<T> {
        self.slm().iter_mut()
    }

    #[inline]
    fn copy_sl( &mut self, x: &[T] ) {
        self.slm().copy_from_slice(x);
    }

    #[inline]
    fn copy<VT: GVec<T>>( &mut self, x: &VT ) {
        self.copy_sl(x.sl());
    }

    #[inline]
    fn reset( &mut self ) {
        self.slm().fill(T::default());
    }

    #[inline]
    fn set( &mut self, val: T ) {
        self.slm().fill(val);
    }

    #[inline]
    fn subvec_mut( &mut self, i1: usize, i2: usize ) -> VecViewMut<T> {
        VecViewMut { view: &mut self.slm()[i1..i2] }
    }

    #[inline]
    fn get_elements<VT: GVec<T>>( &mut self, x: &VT, sel: &[usize] ) {
        for (y_, sel_) in mzip!(self.itm(), sel) {
            *y_ = *x.idx(*sel_);
        }
    }
}

pub trait NVec<T: Numeric>: GVec<T>
{
    #[inline]
    fn sum( &self ) -> T {
        let mut s = T::zero();
        for y_ in self.it() {
            s += y_;
        }
        s
    }
}

pub trait NVecMut<T: Numeric>: NVec<T> + GVecMut<T> 
{
    #[inline]
    fn assign_add<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ + x2_;
        }
    }

    #[inline]
    fn assign_sub<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ - x2_;
        }
    }

    #[inline]
    fn assign_mul<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ * x2_;
        }
    }

    #[inline]
    fn assign_div<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 ) 
    where VT1: NVec<T>, VT2: NVec<T>
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ / x2_;
        }
    }

    #[inline]
    fn addassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ += x_;
        } 
    }

    #[inline]
    fn subassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ -= x_;
        } 
    }

    #[inline]
    fn mulassign<VT>( &mut self, x: &VT ) 
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ *= x_;
        } 
    }

    #[inline]
    fn divassign<VT>( &mut self, x: &VT )
    where VT: NVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ /= x_;
        }
    }

    #[inline]
    fn addassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ += s;
        } 
    }

    #[inline]
    fn subassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ -= s;
        } 
    }

    #[inline]
    fn mulassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ *= s;
        } 
    }

    #[inline]
    fn divassign_scalar( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ /= s;
        } 
    }

    #[inline]
    fn assign_muladd<VT1, VT2, VT3>( &mut self, x: &VT1, a: &VT2, b: &VT3 )
    where VT1: GVec<T>, VT2: GVec<T>, VT3: GVec<T>,
    {
        for elem!(y_, x_, a_, b_) in mzip!(self.itm(), x.it(), a.it(), b.it()) {
            *y_ = x_.mul_add(*a_, *b_);
        }
    }

    #[inline]
    fn muladdassign<VT1, VT2>( &mut self, a: &VT1, b: &VT2 )
    where VT1: GVec<T>, VT2: GVec<T>,
    {
        for elem!(y_, a_, b_) in mzip!(self.itm(), a.it(), b.it()) {
            y_.mul_add_assign(*a_, *b_);
        }
    }
}

pub trait RVec<T: Float>: NVec<T>
{
    #[inline]
    fn sumsquare( &self ) -> T {
        let mut sum: T = T::zero();
        for x_ in self.it() {
            sum += x_.powi(2);
        }
        sum
    }

    #[inline]
    fn norm( &self ) -> T {
        self.sumsquare().sqrt()
    }
}

pub trait RVecMut<T: Float>: RVec<T> + NVecMut<T>
{
    #[inline]
    fn sort_ascend( &mut self ) {
        self.slm().sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    #[inline]
    fn scale( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ *= s;
        } 
    }

    #[inline]
    fn unscale( &mut self, s: T ) {
        for y_ in self.itm() {
            *y_ /= s;
        } 
    }

    #[inline]
    fn assign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = s * x_;
        } 
    }

    #[inline]
    fn assign_powi<VT>( &mut self, x: &VT, n: i32 )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.powi(n);
        } 
    }

    #[inline]
    fn assign_add_scale<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2, s: T )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ + s * x2_;
        } 
    }

    #[inline]
    fn assign_sub_scale<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2, s: T )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = *x1_ - s * x2_;
        } 
    }

    #[inline]
    fn addassign_scale<VT>( &mut self, x: &VT, s: T ) 
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ += s * x_;
        } 
    }

    #[inline]
    fn subassign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ -= s * x_;
        } 
    }

    #[inline]
    fn assign_unscale<VT>( &mut self, x: &VT, s: T )
    where VT: RVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = *x_ / s;
        } 
    }

    #[inline]
    fn assign_absdiff<VT1, VT2>( &mut self, x1: &VT1, x2: &VT2 )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, x1_, x2_) in mzip!(self.itm(), x1.it(), x2.it()) {
            *y_ = (*x1_ - x2_).abs();
        }
    }

    #[inline]
    fn get_complex_norm<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm();
        }
    }

    #[inline]
    fn get_complex_lognorm<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm().ln();
        }
    }

    #[inline]
    fn get_complex_norm2<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.norm_sqr();
        }
    }

    #[inline]
    fn clip<VT1, VT2>( &mut self, lb: &VT1, ub: &VT2 )
    where VT1: RVec<T>, VT2: RVec<T>,
    {
        for elem!(y_, lb_, ub_) in mzip!(self.itm(), lb.it(), ub.it()) {
            if *y_ < *lb_ {
                *y_ = *lb_;
            }
            if *y_ > *ub_ {
                *y_ = *ub_;
            }
        }
    }
    
    #[inline]
    fn clip_lb<VT>( &mut self, lb: &VT )
    where VT: RVec<T>,
    {
        for (y_, lb_) in zip(self.itm(), lb.it()) {
            if *y_ < *lb_ {
                *y_ = *lb_;
            }
        }
    }

    #[inline]
    fn clip_ub<VT>( &mut self, ub: &VT )
    where VT: RVec<T>,
    {
        for (y_, ub_) in zip(self.itm(), ub.it()) {
            if *y_ > *ub_ {
                *y_ = *ub_;
            }
        }
    }
}

pub trait CVec<T: Float>: NVec<Complex<T>> {}

pub trait CVecMut<T: Float>: CVec<T> + NVecMut<Complex<T>> 
{
    #[inline]
    fn scale( &mut self, s: T ) {
        for y_ in self.itm() {
            y_.re *= s;
            y_.im *= s;
        } 
    }

    #[inline]
    fn unscale( &mut self, s: T ) {
        for y_ in self.itm() {
            y_.re /= s;
            y_.im /= s;
        } 
    }

    #[inline]
    fn conj( &mut self ) {
        for y_ in self.itm() {
            y_.im = -y_.im;
        } 
    }

    #[inline]
    fn assign_scale<VT>( &mut self, x: &VT, s: T )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.scale(s);
        } 
    }

    #[inline]
    fn assign_unscale<VT>( &mut self, x: &VT, s: T )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.unscale(s);
        } 
    }

    #[inline]
    fn assign_conj<VT>( &mut self, x: &VT )
    where VT: CVec<T>,
    {
        for (y_, x_) in zip(self.itm(), x.it()) {
            *y_ = x_.conj();
        } 
    }
}

pub trait GMat<T: General>: GVec<T>
{
    fn nrow( &self ) -> usize;

    fn ncol( &self ) -> usize;

    fn stride( &self ) -> usize;

    fn idx2( &self, i: usize, j: usize ) -> &T;

    #[inline]
    fn col( &self, j: usize ) -> VecView<T> {
        let m = self.nrow();
        VecView { view: &self.sl()[j*m..(j+1)*m] }
    }

    #[inline]
    fn cols( &self, j1: usize, j2: usize ) -> MatView<T> {
        let m = self.nrow();
        MatView { dim0: j2-j1, dim1: m, view: &self.sl()[j1*m..j2*m] }
    }

    #[inline]
    fn subvec2( &self, i1: usize, j1: usize, i2: usize, j2: usize ) -> VecView<T> {
        let m = self.nrow();
        VecView { view: &self.sl()[i1+j1*m..i2+j2*m] }
    }

    #[inline]
    fn it_diag( &self ) -> DiagView<T> {
        DiagView { i: 0, dim: self.nrow(), view: self.sl() }
    }

    #[inline]
    fn diag( &self ) -> Vec<T> {
        let mut x: Vec<T> = Vec::with_capacity(self.nrow());
        for i in 0..self.nrow() {
            x.push(*self.idx2(i, i));
        }
        x
    }
}

pub trait GMatMut<T: General>: GMat<T> + GVecMut<T>
{
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T;

    #[inline]
    fn col_mut( &mut self, j: usize ) -> VecViewMut<T> {
        let m = self.nrow();
        VecViewMut { view: &mut self.slm()[j*m..(j+1)*m] }
    }

    #[inline]
    fn cols_mut( &mut self, j1: usize, j2: usize ) -> MatViewMut<T> {
        let m = self.nrow();
        MatViewMut { dim0: j2-j1, dim1: m, view: &mut self.slm()[j1*m..j2*m] }
    }

    #[inline]
    fn subvec2_mut( &mut self, i1: usize, j1: usize, i2: usize, j2: usize ) -> VecViewMut<T> {
        let m = self.nrow();
        VecViewMut { view: &mut self.slm()[i1+j1*m..i2+j2*m] }
    }

    #[inline]
    fn get_rows<MT: GMat<T>>( &mut self, x: &MT, rows: &[usize] ) {
        for j in 0..self.ncol() {
            for (y_, r_) in mzip!(self.col_mut(j).itm(), rows.iter()) {
                *y_ = *x.idx2(*r_, j);
            }
        }
    }

    #[inline]
    fn get_rows_as_cols<MT: GMat<T>>( &mut self, x: &MT, rows: &[usize] ) {
        for (j, r_) in mzip!( 0..self.ncol(), rows.iter() ) {
            for (y_, jx) in mzip!(self.col_mut(j).itm(), 0..x.ncol()) {
                *y_ = *x.idx2(*r_, jx);
            }
        }
    }

    #[inline]
    fn set_diag( &mut self, val: T ) {
        for i in 0..self.ncol() {
            *self.idxm2(i, i) = val;
        }
    }

    #[inline]
    fn set_diag_to_vec<VT: GVec<T>>( &mut self, x: &VT ) {
        for (i, x_) in zip(0..self.ncol(), x.it()) {
            *self.idxm2(i, i) = *x_;
        }
    }

    #[inline]
    fn set_lband( &mut self, offset: usize, val: T ) {
        for j in 0..self.ncol()-offset {
            *self.idxm2(j+offset, j) = val;
        }
    }

    #[inline]
    fn set_uband( &mut self, offset: usize, val: T ) {
        for j in offset..self.ncol() {
            *self.idxm2(j-offset, j) = val;
        }
    }

    #[inline]
    fn set_lower( &mut self, val: T ) {
        for j in 0..self.ncol()-1 {
            for i in j+1..self.nrow() {
                *self.idxm2(i, j) = val;
            }
        }
    }

    #[inline]
    fn set_upper( &mut self, val: T ) {
        for j in 1..self.ncol() {
            for i in 0..j {
                *self.idxm2(i, j) = val;
            }
        }
    }

    #[inline]
    fn copy_lower_to_upper( &mut self ) {
        for j in 1..self.ncol() {
            for i in 0..j {
                *self.idxm2(i, j) = *self.idx2(j, i);
            }
        }
    }

    #[inline]
    fn copy_upper_to_lower( &mut self ) {
        for j in 0..self.ncol()-1 {
            for i in j+1..self.nrow() {
                *self.idxm2(i, j) = *self.idx2(j, i);
            }
        }
    }

    #[inline]
    fn get_trans<MT: GMat<T>>( &mut self, x: &MT ) {
        let nrow = self.nrow();
        for j in 0..self.ncol() {
            for (y_, i) in zip(self.col_mut(j), 0..nrow) {
                *y_ = *x.idx2(j, i);
            }
        }
    }
}

pub trait NMat<T: Numeric>: GMat<T> + NVec<T> {}

pub trait NMatMut<T: Numeric>: NMat<T> + GMatMut<T> + NVecMut<T>
{
    #[inline]
    fn addassign_iden( &mut self, s: T ) {
        for i in 0..self.ncol() {
            *self.idxm2(i, i) += s;
        }
    }
    
    #[inline]
    fn addassign_rowvec<VT: NVec<T>>( &mut self, x: &VT ) {
        for (j, x_) in zip(0..self.ncol(), x.it()) {
            for y_ in self.col_mut(j) {
                *y_ += x_;
            }
        }
    }

    #[inline]
    fn mulassign_rowvec<VT: NVec<T>>( &mut self, x: &VT ) {
        for (j, x_) in zip(0..self.ncol(), x.it()) {
            for y_ in self.col_mut(j) {
                *y_ *= x_;
            }
        }
    }

    #[inline]
    fn muladdassign_rowvec<VT1: NVec<T>, VT2: NVec<T>>( &mut self, a: &VT1, b: &VT2 ) {
        for elem!(j, a_, b_) in mzip!(0..self.ncol(), a.it(), b.it()) {
            for y_ in self.col_mut(j) {
                y_.mul_add_assign(*a_, *b_);
            }
        }
    }
}

pub trait RMat<T: Float>: NMat<T> + RVec<T>
{
    #[inline]
    fn sumlogdiag( &self ) -> T {
        let mut sum: T = T::zero();
        for i in 0..self.ncol() {
            sum += self.idx2(i, i).ln();
        }
        sum
    }
}

pub trait RMatMut<T: Float>: RMat<T> + NMatMut<T> + RVecMut<T> {}

pub trait CMat<T: Float>: NMat<Complex<T>> + CVec<T> {}

pub trait CMatMut<T: Float>: CMat<T> + NMatMut<Complex<T>> + CVecMut<T> {}


impl<T: General, const N: usize> GVec<T> for [T; N]
{
    #[inline]
    fn size( &self ) -> usize {
        N
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.as_slice()
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General, const N: usize> GVecMut<T> for [T; N]
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.as_mut_slice()
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: General, const N: usize> GMat<T> for [T; N]
{
    #[inline]
    fn nrow( &self ) -> usize {
        N
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        N
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General, const N: usize> GMatMut<T> for [T; N]
{
    #[inline]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: Numeric, const N: usize> NVec<T> for [T; N] {}
impl<T: Numeric, const N: usize> NVecMut<T> for [T; N] {}
impl<T: Numeric, const N: usize> NMat<T> for [T; N] {}
impl<T: Numeric, const N: usize> NMatMut<T> for [T; N] {}

impl<T: Float, const N: usize> RVec<T> for [T; N] {}
impl<T: Float, const N: usize> RVecMut<T> for [T; N] {}
impl<T: Float, const N: usize> RMat<T> for [T; N] {}
impl<T: Float, const N: usize> RMatMut<T> for [T; N] {}

impl<T: Float, const N: usize> CVec<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CVecMut<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CMat<T> for [Complex<T>; N] {}
impl<T: Float, const N: usize> CMatMut<T> for [Complex<T>; N] {}


impl<T: General> GVec<T> for Vec<T>
{
    #[inline]
    fn size( &self ) -> usize {
        self.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.as_slice()
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General> GVecMut<T> for Vec<T>
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.as_mut_slice()
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: General> GMat<T> for Vec<T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.len()
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.len()
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.index(i)
    }
}

impl<T: General> GMatMut<T> for Vec<T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.index_mut(i)
    }
}

impl<T: Numeric> NVec<T> for Vec<T> {}
impl<T: Numeric> NVecMut<T> for Vec<T> {}
impl<T: Numeric> NMat<T> for Vec<T> {}
impl<T: Numeric> NMatMut<T> for Vec<T> {}

impl<T: Float> RVec<T> for Vec<T> where T: Float, {}
impl<T: Float> RVecMut<T> for Vec<T> where T: Float, {}
impl<T: Float> RMat<T> for Vec<T> where T: Float, {}
impl<T: Float> RMatMut<T> for Vec<T> where T: Float, {}

impl<T: Float> CVec<T> for Vec<Complex<T>> {}
impl<T: Float> CVecMut<T> for Vec<Complex<T>> {}
impl<T: Float> CMat<T> for Vec<Complex<T>> {}
impl<T: Float> CMatMut<T> for Vec<Complex<T>> {}


impl<'a, T: General> GVec<T> for VecView<'a, T>
{
    #[inline]
    fn size( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.view.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.view
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GVec<T> for VecViewMut<'a, T>
{
    #[inline]
    fn size( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.view.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.view
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GVecMut<T> for VecViewMut<'a, T>
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.view.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.view
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.view.index_mut(i)
    }
}

impl<'a, T: General> GMat<T> for VecView<'a, T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GMat<T> for VecViewMut<'a, T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GMatMut<T> for VecViewMut<'a, T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.view.index_mut(i)
    }
}

impl<'a, T: Numeric> NVec<T> for VecView<'a, T> {}
impl<'a, T: Numeric> NVec<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NVecMut<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NMat<T> for VecView<'a, T> {}
impl<'a, T: Numeric> NMat<T> for VecViewMut<'a, T> {}
impl<'a, T: Numeric> NMatMut<T> for VecViewMut<'a, T> {}

impl<'a, T: Float> RVec<T> for VecView<'a, T> {}
impl<'a, T: Float> RVec<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RVecMut<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RMat<T> for VecView<'a, T> {}
impl<'a, T: Float> RMat<T> for VecViewMut<'a, T> {}
impl<'a, T: Float> RMatMut<T> for VecViewMut<'a, T> {}

impl<'a, T: Float> CVec<T> for VecView<'a, Complex<T>> {}
impl<'a, T: Float> CVec<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CVecMut<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for VecView<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for VecViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMatMut<T> for VecViewMut<'a, Complex<T>> {}


impl<'a, T: General> GVec<T> for MatView<'a, T>
{
    #[inline]
    fn size( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.view.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.view
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GVec<T> for MatViewMut<'a, T>
{
    #[inline]
    fn size( &self ) -> usize {
        self.view.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.view.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.view
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.view.index(i)
    }
}

impl<'a, T: General> GVecMut<T> for MatViewMut<'a, T>
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.view.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.view
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.view.index_mut(i)
    }
}

impl<'a, T: General> GMat<T> for MatView<'a, T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn ncol( &self ) -> usize {
        self.dim0
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.view.index(i+j*self.dim1)
    }
}

impl<'a, T: General> GMat<T> for MatViewMut<'a, T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn ncol( &self ) -> usize {
        self.dim0
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.dim1
    }

    #[inline]
    fn idx2( &self, i: usize, j: usize ) -> &T {
        self.view.index(i+j*self.dim1)
    }
}

impl<'a, T: General> GMatMut<T> for MatViewMut<'a, T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        self.view.index_mut(i+j*self.dim1)
    }
}

impl<'a, T: Numeric> NVec<T> for MatView<'a, T> {}
impl<'a, T: Numeric> NVec<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NVecMut<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NMat<T> for MatView<'a, T> {}
impl<'a, T: Numeric> NMat<T> for MatViewMut<'a, T> {}
impl<'a, T: Numeric> NMatMut<T> for MatViewMut<'a, T> {}

impl<'a, T: Float> RVec<T> for MatView<'a, T> {}
impl<'a, T: Float> RVec<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RVecMut<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RMat<T> for MatView<'a, T> {}
impl<'a, T: Float> RMat<T> for MatViewMut<'a, T> {}
impl<'a, T: Float> RMatMut<T> for MatViewMut<'a, T> {}

impl<'a, T: Float> CVec<T> for MatView<'a, Complex<T>> {}
impl<'a, T: Float> CVec<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CVecMut<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for MatView<'a, Complex<T>> {}
impl<'a, T: Float> CMat<T> for MatViewMut<'a, Complex<T>> {}
impl<'a, T: Float> CMatMut<T> for MatViewMut<'a, Complex<T>> {}


impl<'a, T> IntoIterator for VecView<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.view.into_iter()
    }
}

impl<'a, T> IntoIterator for VecViewMut<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.view.into_iter()
    }
}

impl<'a, T> IntoIterator for MatView<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.view.into_iter()
    }
}

impl<'a, T> IntoIterator for MatViewMut<'a, T>
where T: Default + Copy + Clone,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.view.into_iter()
    }
}


pub struct RVecPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct CVecPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct RMatPrinter {
    pub width: usize,
    pub prec: usize,
}

pub struct CMatPrinter {
    pub width: usize,
    pub prec: usize,
}

impl RVecPrinter
{
    #[inline]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline]
    pub fn print<T: Float + Display, VT: RVec<T>>( &self, name: &str, x: &VT ) {
        let width = self.width;
        let prec = self.prec;
        print!("{name} = \n(");
        for x_ in x.it() {
            print!("{x_:>width$.prec$},");
        }
        println!(")");
    }
}

impl CVecPrinter
{
    #[inline]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline]
    pub fn print<T: Float + Display, VT: CVec<T>>( &self, name: &str, x: &VT ) {
        let width = self.width;
        let width2 = width - 1;
        let prec = self.prec;
        print!("{name} =\n(");
        for x_ in x.it() {
            print!("{:>width$.prec$} {:>+width2$.prec$}j,", x_.re, x_.im);
        }
        println!(")");
    }
}

impl RMatPrinter
{
    #[inline]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline]
    pub fn print<T: Float + Display, MT: RMat<T>>( &self, name: &str, x: &MT ) {
        let width = self.width;
        let prec = self.prec;
        print!("{name} =");
        for i in 0..x.nrow() {
            print!("\n(");
            for j in 0..x.ncol() {
                print!("{:>width$.prec$},", x.idx2(i,j));
            }
            print!(")");
        }
        println!();
    }
}

impl CMatPrinter
{
    #[inline]
    pub fn new( width: usize, prec: usize ) -> Self {
        Self { width, prec }
    }

    #[inline]
    pub fn print<T: Float + Display, MT: CMat<T>>( &self, name: &str, x: &MT ) {
        let width = self.width;
        let width2 = width - 1;
        let prec = self.prec;
        print!("{name} =");
        for i in 0..x.nrow() {
            print!("\n(");
            for j in 0..x.ncol() {
                let x_ = x.idx2(i,j);
                print!("{:>width$.prec$} {:>+width2$.prec$}j,", x_.re, x_.im);
            }
            print!(")");
        }
        println!();
    }
}