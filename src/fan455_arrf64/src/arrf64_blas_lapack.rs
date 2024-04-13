use super::arrf64_basic::*;

type BlasInt = i32;
type BlasChar = i8;

pub const NOTRANS: i8 = 78_i8;
pub const TRANS: i8 = 84_i8;
pub const UPPER: i8 = 85_i8;
pub const LOWER: i8 = 76_i8;
pub const UNIT: i8 = 85_i8;
pub const NONUNIT: i8 = 78_i8;
pub const LEFT: i8 = 76_i8;
pub const RIGHT: i8 = 82_i8;

extern "C" {

pub fn dnrm2_(
    n: *const BlasInt,
    x: *const f64,
    incx: *const BlasInt,
) -> f64;

pub fn ddot_(
    n: *const BlasInt,
    x: *const f64,
    incx: *const BlasInt,
    y: *const f64,
    incy: *const BlasInt
) -> f64;

pub fn daxpy_(
    n: *const BlasInt,
    a: *const f64,
    x: *const f64,
    incx: *const BlasInt,
    y: *mut f64,
    incy: *const BlasInt
); 

pub fn daxpby_(
    n: *const BlasInt,
    a: *const f64,
    x: *const f64,
    incx: *const BlasInt,
    b: *const f64,
    y: *mut f64,
    incy: *const BlasInt
);

pub fn dgemv_(
    trans: *const BlasChar,
    m: *const BlasInt,
    n: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    x: *const f64,
    incx: *const BlasInt,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt
);

pub fn dsymv_(
    uplo: *const BlasChar,
    n: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    x: *const f64,
    incx: *const BlasInt,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt
);

pub fn dtrmv_(
    uplo: *const BlasChar,
    trans: *const BlasChar,
    diag: *const BlasChar,
    n: *const BlasInt,
    a: *const f64,
    lda: *const BlasInt,
    x: *mut f64,
    incx: *const BlasInt
);

pub fn dger_(
    m: *const BlasInt,
    n: *const BlasInt,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt,
    y: *const f64,
    incy: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt
);

pub fn dsyr_(
    uplo: *const BlasChar,
    n: *const BlasInt,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt
);

pub fn dsymm_(
    side: *const BlasChar,
    uplo: *const BlasChar,
    m: *const BlasInt,
    n: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    b: *const f64,
    ldb: *const BlasInt,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt
);

pub fn dgemm_(
    transa: *const BlasChar,
    transb: *const BlasChar,
    m: *const BlasInt,
    n: *const BlasInt,
    k: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    b: *const f64,
    ldb: *const BlasInt,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt
);

pub fn dsyrk_(
    uplo: *const BlasChar,
    trans: *const BlasChar,
    n: *const BlasInt,
    k: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt
);

pub fn dtrmm_(
    side: *const BlasChar,
    uplo: *const BlasChar,
    transa: *const BlasChar,
    diag: *const BlasChar,
    m: *const BlasInt,
    n: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    b: *mut f64,
    ldb: *const BlasInt
);

pub fn dtrsm_(
    side: *const BlasChar,
    uplo: *const BlasChar,
    transa: *const BlasChar,
    diag: *const BlasChar,
    m: *const BlasInt,
    n: *const BlasInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt,
    b: *mut f64,
    ldb: *const BlasInt
);

pub fn dpotrf_(
    uplo: *const BlasChar,
    n: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt,
    info: *mut BlasInt
);

pub fn dpotri_(
    uplo: *const BlasChar,
    n: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt,
    info: *mut BlasInt
);

pub fn dposv_(
    uplo: *const BlasChar,
    n: *const BlasInt,
    nrhs: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt,
    b: *mut f64,
    ldb: *const BlasInt,
    info: *mut BlasInt
);

pub fn dgesv_(
    n: *const BlasInt,
    nrhs: *const BlasInt,
    a: *mut f64,
    lda: *const BlasInt,
    ipiv: *mut BlasInt,
    b: *mut f64,
    ldb: *const BlasInt,
    info: *mut BlasInt
);

pub fn idamax_(
    n: *const BlasInt,
    x: *const f64, 
    incx: *const BlasInt
) -> BlasInt;

}


#[inline]
pub fn idamax<VT: RVec<f64>>(
    x: &VT
) -> usize {
    let n: BlasInt = x.size() as BlasInt;
    let incx: BlasInt = 1;
    unsafe { (idamax_(
        &n as *const BlasInt,
        x.ptr(),
        &incx as *const BlasInt
    ) - 1) as usize }
}


#[inline]
pub fn dgesv<MT1: RMatMut<f64>, MT2: RMatMut<f64>>(
    a: &mut MT1,
    b: &mut MT2,
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let nrhs: BlasInt = b.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    let mut info: BlasInt = 0_i32;
    let mut ipiv: Vec<BlasInt> = vec![0; a.nrow()];
    unsafe { dgesv_(
        &n as *const BlasInt,
        &nrhs as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt,
        ipiv.as_mut_ptr(),
        b.ptrm(),
        &ldb as *const BlasInt,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dposv<MT1: RMatMut<f64>, MT2: RMatMut<f64>>(
    a: &mut MT1,
    b: &mut MT2,
    uplo: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let nrhs: BlasInt = b.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    let mut info: BlasInt = 0_i32;
    unsafe { dposv_(
        &uplo as *const BlasChar,
        &n as *const BlasInt,
        &nrhs as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt,
        b.ptrm(),
        &ldb as *const BlasInt,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dpotri<MT: RMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let mut info: BlasInt = 0_i32;
    unsafe { dpotri_(
        &uplo as *const BlasChar,
        &n as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dpotrf<MT: RMatMut<f64>>(
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let mut info: BlasInt = 0_i32;
    unsafe { dpotrf_(
        &uplo as *const BlasChar,
        &n as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt,
        &mut info as *mut BlasInt
    ); }
}


#[inline]
pub fn dtrmm<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &mut MT2,
    side: BlasChar,
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let m: BlasInt = b.nrow() as BlasInt;
    let n: BlasInt = b.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    unsafe { dtrmm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        b.ptrm(),
        &ldb as *const BlasInt,
    ); }
}


#[inline]
pub fn dtrsm<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &mut MT2,
    side: BlasChar,
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let m: BlasInt = b.nrow() as BlasInt;
    let n: BlasInt = b.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    unsafe { dtrsm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        b.ptrm(),
        &ldb as *const BlasInt,
    ); }
}


#[inline]
pub fn dsyrk_notrans<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    beta: f64, 
    c: &mut MT2,
    trans: BlasChar,
    uplo: BlasChar
) {
    let n: BlasInt = c.nrow() as BlasInt;
    let k: BlasInt = a.ncol() as BlasInt; // difference here
    let lda: BlasInt = a.stride() as BlasInt;
    let ldc: BlasInt = c.stride() as BlasInt;
    unsafe { dsyrk_(
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &n as *const BlasInt,
        &k as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasInt,
    ); }
}


#[inline]
pub fn dsyrk_trans<MT1: RMat<f64>, MT2: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    beta: f64, 
    c: &mut MT2,
    trans: BlasChar,
    uplo: BlasChar
) {
    let n: BlasInt = c.nrow() as BlasInt;
    let k: BlasInt = a.nrow() as BlasInt; // difference here
    let lda: BlasInt = a.stride() as BlasInt;
    let ldc: BlasInt = c.stride() as BlasInt;
    unsafe { dsyrk_(
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &n as *const BlasInt,
        &k as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasInt,
    ); }
}


#[inline]
pub fn dgemm_notransa<MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &MT2,
    beta: f64, 
    c: &mut MT3,
    transa: BlasChar,
    transb: BlasChar
) {
    let m: BlasInt = c.nrow() as BlasInt;
    let n: BlasInt = c.ncol() as BlasInt;
    let k: BlasInt = a.ncol() as BlasInt; // difference here
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    let ldc: BlasInt = c.stride() as BlasInt;
    unsafe { dgemm_(
        &transa as *const BlasChar,
        &transb as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &k as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        b.ptr(),
        &ldb as *const BlasInt,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasInt,
    ); }
}


#[inline]
pub fn dgemm_transa<MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &MT2,
    beta: f64, 
    c: &mut MT3,
    transa: BlasChar,
    transb: BlasChar
) {
    let m: BlasInt = c.nrow() as BlasInt;
    let n: BlasInt = c.ncol() as BlasInt;
    let k: BlasInt = a.nrow() as BlasInt; // difference here
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    let ldc: BlasInt = c.stride() as BlasInt;
    unsafe { dgemm_(
        &transa as *const BlasChar,
        &transb as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &k as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        b.ptr(),
        &ldb as *const BlasInt,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasInt,
    ); }
}


#[inline]
pub fn dsymm<MT1: RMat<f64>, MT2: RMat<f64>, MT3: RMatMut<f64>>( 
    alpha: f64,   
    a: &MT1,
    b: &MT2,
    beta: f64, 
    c: &mut MT3,
    side: BlasChar,
    uplo: BlasChar
) {
    let m: BlasInt = c.nrow() as BlasInt;
    let n: BlasInt = c.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let ldb: BlasInt = b.stride() as BlasInt;
    let ldc: BlasInt = c.stride() as BlasInt;
    unsafe { dsymm_(
        &side as *const BlasChar,
        &uplo as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        b.ptr(),
        &ldb as *const BlasInt,
        &beta as *const f64,
        c.ptrm(),
        &ldc as *const BlasInt,
    ); }
}


#[inline]
pub fn dsyr<VT: RVec<f64>, MT: RMatMut<f64>>( 
    alpha: f64, 
    x: &VT,  
    a: &mut MT,
    uplo: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let incx: BlasInt = 1;
    unsafe { dsyr_(
        &uplo as *const BlasChar,
        &n as *const BlasInt,
        &alpha as *const f64,
        x.ptr(),
        &incx as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt
    ); }
}


#[inline]
pub fn dger<VT1: RVec<f64>, VT2: RVec<f64>, MT: RMatMut<f64>>( 
    alpha: f64, 
    x: &VT1,  
    y: &VT2,
    a: &mut MT
) {
    let m: BlasInt = a.nrow() as BlasInt;
    let n: BlasInt = a.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { dger_(
        &m as *const BlasInt,
        &n as *const BlasInt,
        &alpha as *const f64,
        x.ptr(),
        &incx as *const BlasInt,
        y.ptr(),
        &incy as *const BlasInt,
        a.ptrm(),
        &lda as *const BlasInt
    ); }
}


#[inline]
pub fn dtrmv<VT: RVecMut<f64>, MT: RMat<f64>>( 
    a: &MT, 
    x: &mut VT, 
    trans: BlasChar,
    uplo: BlasChar,
    diag: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let incx: BlasInt = 1;
    unsafe { dtrmv_(
        &uplo as *const BlasChar,
        &trans as *const BlasChar,
        &diag as *const BlasChar,
        &n as *const BlasInt,
        a.ptr(),
        &lda as *const BlasInt,
        x.ptrm(),
        &incx as *const BlasInt
    ); }
}


#[inline]
pub fn ddot<VT1: RVec<f64>, VT2: RVec<f64>>( 
    x: &VT1, 
    y: &VT2 
) -> f64 {
    let n: BlasInt = x.size() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { ddot_(
        &n as *const BlasInt, 
        x.ptr(),
        &incx as *const BlasInt,
        y.ptr(),
        &incy as *const BlasInt
    ) }
}

#[inline]
pub fn dnrm2<VT: RVec<f64>>( 
    x: &VT, 
) -> f64 {
    let n: BlasInt = x.size() as BlasInt;
    let incx: BlasInt = 1;
    unsafe { dnrm2_(
        &n as *const BlasInt, 
        x.ptr(),
        &incx as *const BlasInt,
    ) }
}

#[inline]
pub fn daxpy<VT1: RVec<f64>, VT2: RVecMut<f64>>( 
    a: f64, 
    x: &VT1, 
    y: &mut VT2
) {
    let n: BlasInt = x.size() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { daxpy_(
        &n as *const BlasInt, 
        &a as *const f64,
        x.ptr(),
        &incx as *const BlasInt,
        y.ptrm(),
        &incy as *const BlasInt
    ); }
}

#[inline]
pub fn daxpby<VT1: RVec<f64>, VT2: RVecMut<f64>>( 
    a: f64, 
    x: &VT1,
    b: f64,
    y: &mut VT2
) {
    let n: BlasInt = x.size() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { daxpby_(
        &n as *const BlasInt, 
        &a as *const f64,
        x.ptr(),
        &incx as *const BlasInt,
        &b as *const f64,
        y.ptrm(),
        &incy as *const BlasInt
    ); }
}

#[inline]
pub fn dgemv<VT1: RVec<f64>, VT2: RVecMut<f64>, MT: RMat<f64>>( 
    alpha: f64, 
    a: &MT, 
    x: &VT1, 
    beta: f64, 
    y: &mut VT2, 
    trans: BlasChar 
) {
    let m: BlasInt = a.nrow() as BlasInt;
    let n: BlasInt = a.ncol() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { dgemv_(
        &trans as *const BlasChar,
        &m as *const BlasInt,
        &n as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        x.ptr(),
        &incx as *const BlasInt,
        &beta as *const f64,
        y.ptrm(),
        &incy as *const BlasInt
    ); }
}


#[inline]
pub fn dsymv<VT1: RVec<f64>, VT2: RVecMut<f64>, MT: RMat<f64>>( 
    alpha: f64, 
    a: &MT, 
    x: &VT1, 
    beta: f64, 
    y: &mut VT2,
    uplo: BlasChar
) {
    let n: BlasInt = a.nrow() as BlasInt;
    let lda: BlasInt = a.stride() as BlasInt;
    let incx: BlasInt = 1;
    let incy: BlasInt = 1;
    unsafe { dsymv_(
        &uplo as *const BlasChar,
        &n as *const BlasInt,
        &alpha as *const f64,
        a.ptr(),
        &lda as *const BlasInt,
        x.ptr(),
        &incx as *const BlasInt,
        &beta as *const f64,
        y.ptrm(),
        &incy as *const BlasInt
    ); }
}




