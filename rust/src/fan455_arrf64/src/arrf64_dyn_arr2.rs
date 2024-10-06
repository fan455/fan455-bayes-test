use super::arrf64_basic::*;
use std::ops::{Index, IndexMut};
use std::iter::IntoIterator;
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};
use fan455_util::*;


#[derive(Default, Debug)]
pub struct Mat<T: General>
{
    pub dim0: usize,
    pub dim1: usize,
    pub data: Vec<T>,
}

impl<T: General> Mat<T>
{
    #[inline]
    pub fn new( m: usize, n: usize ) -> Self {
        let data: Vec<T> = vec![T::default(); m*n];
        Self { dim0: n, dim1: m, data }
    }

    #[inline]
    pub fn new_empty() -> Self {
        Self { dim0: 0, dim1: 0, data: Vec::<T>::new() }
    }

    #[inline]
    pub fn new_set( m: usize, n: usize, val: T ) -> Self {
        let data: Vec<T> = vec![val; m*n];
        Self { dim0: n, dim1: m, data }
    }

    #[inline]
    pub fn new_copy<MT: GMat<T>>( x: &MT ) -> Self {
        let dim0 = x.ncol();
        let dim1 = x.nrow();
        let mut data: Vec<T> = Vec::with_capacity(x.size());
        data.extend_from_slice(x.sl());
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn from_vec( x: Vec<T>, m: usize, n: usize ) -> Self {
        Self { dim0: n, dim1: m, data: x }
    }

    #[inline]
    pub fn reshape( &mut self, m: usize, n: usize ) {
        let size = self.data.len();
        if m*n != size {
            panic!("reshape matrix to m = {m}, n = {n} fails because data size is {size}");
        }
        self.dim0 = n;
        self.dim1 = m;
    }

    #[inline]
    pub fn clear( &mut self ) {
        self.data.resize(0, T::default());
        self.data.shrink_to_fit();
    }

    #[inline]
    pub fn resize( &mut self, m: usize, n: usize, val: T ) {
        self.reshape(m, n);
        self.data.resize(m*n, val);
    }

    #[inline]
    pub fn truncate( &mut self, m: usize, n: usize ) {
        self.reshape(m, n);
        self.data.truncate(m*n);
        self.data.shrink_to_fit();
    }

    #[inline]
    pub fn is_scalar( &mut self ) -> bool {
        self.data.len() == 1
    }

    #[inline]
    pub fn ensure_row_vec( &mut self ) {
        // If self is column vector, reshape it as a row vector. Otherwise do nothing.
        if self.is_scalar() {
        } else if self.nrow() == 1 {
        } else if self.ncol() == 1 {
            let k = self.nrow();
            self.reshape(1, k);
        }
    }

    #[inline]
    pub fn ensure_col_vec( &mut self ) {
        // If self is row vector, reshape it as a column vector. Otherwise do nothing.
        if self.is_scalar() {
        } else if self.ncol() == 1 {
        } else if self.nrow() == 1 {
            let k = self.ncol();
            self.reshape(k, 1);
        }
    }

    #[inline]
    pub fn view( &self ) -> MatView<T> {
        MatView::new(self.nrow(), self.ncol(), self.data.sl())
    }

    #[inline]
    pub fn view_mut( &mut self ) -> MatViewMut<T> {
        let m = self.nrow();
        let n = self.ncol();
        MatViewMut::new(m, n, self.data.slm())
    }
}


impl<T> Mat<T>
where T: NpyDescrGetter+Default+Copy, NpyObject<T>: NpyTrait<T>
{
    #[inline]
    pub fn read_npy( path: &str ) -> Self {
        let mut obj = NpyObject::<T>::new_reader(path);
        obj.read_header().unwrap();
        let data = obj.read();
        let (dim0, dim1) = match obj.fortran_order {
            false => (obj.shape[0], obj.shape[1]),
            true => (obj.shape[1], obj.shape[0]),
        };
        Self { dim0, dim1, data }
    }

    #[inline]
    pub fn write_npy( &self, path: &str ) {
        let mut obj = NpyObject::<T>::new_writer(path, [1, 0], true, vec![self.nrow(), self.ncol()]);
        obj.write_header().unwrap();
        obj.write(&self.data);
    }
}


impl<T: General> IntoIterator for Mat<T>
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a Mat<T>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a mut Mat<T>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}

impl<T: General> GMatAlloc<T> for Mat<T>
{
    #[inline]
    fn alloc( m: usize, n: usize ) -> Self {
        Self::new(m, n)
    }

    #[inline]
    fn alloc_set( m: usize,n: usize, val: T ) -> Self {
        Self::new_set(m, n, val)
    }

    #[inline]
    fn alloc_copy<VT: GMat<T>>( x: &VT ) -> Self {
        Self::new_copy(x)
    }
}

impl<T: General> GVec<T> for Mat<T>
{
    #[inline]
    fn size ( &self ) -> usize {
        self.data.len()
    }

    #[inline]
    fn ptr( &self ) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn sl( &self ) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    fn idx( &self, i: usize ) -> &T {
        self.data.index(i)
    }
}

impl<T: General> GVecMut<T> for Mat<T>
{
    #[inline]
    fn ptrm( &mut self ) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    fn slm( &mut self ) -> &mut [T] {
        self.data.as_mut_slice()
    }

    #[inline]
    fn idxm( &mut self, i: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<T: General> GMat<T> for Mat<T>
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
        self.data.index(i+j*self.dim1)
    }

    #[inline]
    fn idx2_unchecked( &self, i: usize, j: usize ) -> &T {
        unsafe { self.data.get_unchecked(i+j*self.dim1) }
    }
}

impl<T: General> GMatMut<T> for Mat<T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, j: usize ) -> &mut T {
        self.data.index_mut(i+j*self.dim1)
    }

    #[inline]
    fn idxm2_unchecked( &mut self, i: usize, j: usize ) -> &mut T {
        unsafe { self.data.get_unchecked_mut(i+j*self.dim1) }
    }
}

impl<T: Numeric> NVec<T> for Mat<T> {}
impl<T: Numeric> NVecMut<T> for Mat<T> {}
impl<T: Numeric> NMat<T> for Mat<T> {}
impl<T: Numeric> NMatMut<T> for Mat<T> {}

impl<T: Float> RVec<T> for Mat<T> {}
impl<T: Float> RVecMut<T> for Mat<T> {}
impl<T: Float> RMat<T> for Mat<T> {}
impl<T: Float> RMatMut<T> for Mat<T> {}

impl<T: Float> CVec<T> for Mat<Complex<T>> {}
impl<T: Float> CVecMut<T> for Mat<Complex<T>> {}
impl<T: Float> CMat<T> for Mat<Complex<T>> {}
impl<T: Float> CMatMut<T> for Mat<Complex<T>> {}


impl<T: General> Index<(usize, usize)> for Mat<T>
{
    type Output = T;

    #[inline]
    fn index( &self, index: (usize, usize) ) -> &T {
        self.data.index(index.0+index.1*self.dim1)
    }
}

impl<T: General> IndexMut<(usize, usize)> for Mat<T>
{
    #[inline]
    fn index_mut( &mut self, index: (usize, usize) ) -> &mut T {
        self.data.index_mut(index.0+index.1*self.dim1)
    }
}

impl<T: General> Clone for Mat<T>
{
    fn clone(&self) -> Self {
        Self { dim0: self.dim0, dim1: self.dim1, data: self.data.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}
