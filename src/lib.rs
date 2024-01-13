/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod conversion;
mod enums;
mod interface_mpi;
pub use crate::enums::*;
pub use crate::interface_mpi::*;
