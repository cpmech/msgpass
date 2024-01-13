#[derive(Clone, Copy)]
pub(crate) enum MpiType {
    I32 = 0,
    I64 = 1,
    U32 = 2,
    U64 = 3,
    F32 = 4,
    F64 = 5,
}

#[derive(Clone, Copy)]
pub enum MpiOp {
    Max = 0,  // maximum
    Min = 1,  // minimum
    Sum = 2,  // sum
    Prod = 3, // product
    Land = 4, // logical and
    Band = 5, // bit-wise and
    Lor = 6,  // logical or
    Bor = 7,  // bit-wise or
    Lxor = 8, // logical xor
    Bxor = 9, // bit-wise xor
}

impl MpiType {
    pub fn n(&self) -> i32 {
        *self as i32
    }
}

impl MpiOp {
    pub fn n(&self) -> i32 {
        *self as i32
    }
}
