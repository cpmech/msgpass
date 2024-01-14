#[derive(Clone, Copy)]
pub(crate) enum MpiType {
    I32 = 0,
    I64 = 1,
    U32 = 2,
    U64 = 3,
    F32 = 4,
    F64 = 5,
}

/// Specifies the MPI operator used in reduce-like functions (for integer arrays)
#[derive(Clone, Copy)]
pub enum MpiOp {
    Max = 0,  // maximum
    Min = 1,  // minimum
    Sum = 2,  // sum
    Prod = 3, // product
    Land = 4, // logical and
    Lor = 5,  // logical or
    Lxor = 6, // logical xor
}

/// Specifies the MPI operator used in reduce-like functions (for float number arrays)
#[derive(Clone, Copy)]
pub enum MpiOpx {
    Max = 0,  // maximum
    Min = 1,  // minimum
    Sum = 2,  // sum
    Prod = 3, // product
}

impl MpiType {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOp {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOpx {
    pub(crate) fn n(&self) -> i32 {
        *self as i32
    }
}
