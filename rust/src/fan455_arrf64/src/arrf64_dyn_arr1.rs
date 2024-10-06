use super::arrf64_basic::*;
use std::iter::IntoIterator;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};
use num_complex::Complex;
use fan455_math_scalar::{General, Numeric, Float};
use fan455_util::*;


#[derive(Default, Debug)]
pub struct Arr<T: General>
{
    pub data: Vec<T>,
}


impl<T: General> Arr<T>
{
    #[inline]
    pub fn new( n: usize ) -> Self {
        let data: Vec<T> = vec![T::default(); n];
        Self { data }
    }

    #[inline]
    pub fn new_empty() -> Self {
        Self { data: Vec::<T>::new() }
    }

    #[inline]
    pub fn new_set( n: usize, val: T ) -> Self {
        let data: Vec<T> = vec![val; n];
        Self { data }
    }

    #[inline]
    pub fn new_copy<VT: GVec<T>>( x: &VT ) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(x.size());
        data.extend_from_slice(x.sl());
        Self { data }
    }

    #[inline]
    pub fn from_vec( x: Vec<T> ) -> Self {
        Self { data: x }
    }

    #[inline]
    pub fn resize( &mut self, n: usize, val: T ) {
        self.data.resize(n, val);
    }

    #[inline]
    pub fn truncate( &mut self, n: usize ) {
        self.data.truncate(n);
        self.data.shrink_to_fit();
    }

    #[inline]
    pub fn clear( &mut self ) {
        self.data.resize(0, T::default());
        self.data.shrink_to_fit();
    }
}


impl<T> Arr<T>
where T: NpyDescrGetter+Default+Copy, NpyObject<T>: NpyTrait<T>
{
    #[inline]
    pub fn read_npy( path: &str ) -> Self {
        let mut obj = NpyObject::<T>::new_reader(path);
        obj.read_header().unwrap();
        let data = obj.read();
        Self { data }
    }

    #[inline]
    pub fn write_npy( &self, path: &str ) {
        let mut obj = NpyObject::<T>::new_writer(path, [1, 0], false, vec![self.size()]);
        obj.write_header().unwrap();
        obj.write(&self.data);
    }
}


impl<T: General> IntoIterator for Arr<T>
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a Arr<T>
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&self.data).into_iter()
    }
}

impl<'a, T: General> IntoIterator for &'a mut Arr<T>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    
    #[inline]
    fn into_iter( self ) -> Self::IntoIter {
        (&mut self.data).into_iter()
    }
}


impl<T: General> GVecAlloc<T> for Arr<T>
{
    #[inline]
    fn alloc( n: usize ) -> Self {
        Self::new(n)
    }

    #[inline]
    fn alloc_set( n: usize, val: T ) -> Self {
        Self::new_set(n, val)
    }

    #[inline]
    fn alloc_copy<VT: GVec<T>>( x: &VT ) -> Self {
        Self::new_copy(x)
    }
}


impl<T: General> GVec<T> for Arr<T>
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

impl<T: General> GVecMut<T> for Arr<T>
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

impl<T: General> GMat<T> for Arr<T>
{
    #[inline]
    fn nrow( &self ) -> usize {
        self.data.len()
    }

    #[inline]
    fn ncol( &self ) -> usize {
        1
    }

    #[inline]
    fn stride( &self ) -> usize {
        self.data.len()
    }

    #[inline]
    fn idx2( &self, i: usize, _j: usize ) -> &T {
        self.data.index(i)
    }
}

impl<T: General> GMatMut<T> for Arr<T>
{
    #[inline]
    fn idxm2( &mut self, i: usize, _j: usize ) -> &mut T {
        self.data.index_mut(i)
    }
}

impl<T: Numeric> NVec<T> for Arr<T> {}
impl<T: Numeric> NVecMut<T> for Arr<T> {}
impl<T: Numeric> NMat<T> for Arr<T> {}
impl<T: Numeric> NMatMut<T> for Arr<T> {}

impl<T: Float> RVec<T> for Arr<T> {}
impl<T: Float> RVecMut<T> for Arr<T> {}
impl<T: Float> RMat<T> for Arr<T> {}
impl<T: Float> RMatMut<T> for Arr<T> {}

impl<T: Float> CVec<T> for Arr<Complex<T>> {}
impl<T: Float> CVecMut<T> for Arr<Complex<T>> {}
impl<T: Float> CMat<T> for Arr<Complex<T>> {}
impl<T: Float> CMatMut<T> for Arr<Complex<T>> {}


impl<T: General> Index<usize> for Arr<T>
{
    type Output = T;

    #[inline]
    fn index( &self, index: usize ) -> &T {
        self.data.index(index)
    }
}

impl<T: General> IndexMut<usize> for Arr<T>
{
    #[inline]
    fn index_mut( &mut self, index: usize ) -> &mut T {
        self.data.index_mut(index)
    }
}

impl<T: General> Clone for Arr<T>
{
    fn clone( &self ) -> Self {
        Self { data: self.data.clone() }
    }

    fn clone_from( &mut self, source: &Self ) {
        self.data.clone_from(&source.data);
    }
}


