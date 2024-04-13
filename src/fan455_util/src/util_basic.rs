#[macro_export]
macro_rules! elem {
    ($x:ident) => {
        ($x)
    };
    ($x:ident, $y:ident) => {
        ($x, $y)
    };
    ($x:ident, $y:ident, $($rest:ident),+) => {
        ($x, elem!($y, $($rest),+))
    };
}

#[macro_export]
macro_rules! mzip {
    ($x:ident) => {
        ($x)
    };
    ($x:expr, $y:expr) => {
        std::iter::zip($x, $y)
    };
    ($x:expr, $y:expr, $($rest:expr),+) => {
        std::iter::zip($x, mzip!($y, $($rest),+))
    };
}

#[macro_export]
macro_rules! assert_multi_eq {
    ($first:expr, $($other:expr),+) => {
        $(
            assert_eq!($first, $other);
        )+
    };
}