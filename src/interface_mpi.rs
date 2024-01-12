extern "C" {
    fn c_mpi_initialized() -> i32;
    fn c_mpi_init();
    fn c_mpi_finalize();
}

pub fn mpi_init() {
    unsafe {
        c_mpi_init();
    }
}

pub fn mpi_finalize() {
    unsafe {
        c_mpi_finalize();
    }
}

pub fn mpi_initialized() -> bool {
    unsafe {
        let res = c_mpi_initialized();
        if res == 1 {
            return true;
        } else {
            return false;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mpi_finalize, mpi_init, mpi_initialized};

    #[test]
    fn essential_functions_work() {
        assert!(!mpi_initialized());
        mpi_init();
        assert!(mpi_initialized());
        mpi_finalize();
    }
}
