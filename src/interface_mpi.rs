extern "C" {
    fn c_mpi_initialized() -> i32;
    fn c_mpi_init();
    fn c_mpi_finalize();
    fn c_mpi_world_rank() -> i32;
    fn c_mpi_world_size() -> i32;
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

pub fn mpi_world_rank() -> usize {
    unsafe { c_mpi_world_rank() as usize }
}

pub fn mpi_world_size() -> usize {
    unsafe { c_mpi_world_size() as usize }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn essential_functions_work() {
        assert!(!mpi_initialized());
        mpi_init();
        assert!(mpi_initialized());
        assert_eq!(mpi_world_rank(), 0);
        assert_eq!(mpi_world_size(), 1);
        mpi_finalize();
    }
}
