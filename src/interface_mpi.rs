use crate::constants::*;
use crate::StrError;

extern "C" {
    fn c_mpi_init() -> i32;
    fn c_mpi_init_threaded() -> i32;
    fn c_mpi_finalize() -> i32;
    fn c_mpi_initialized(flag: *mut i32) -> i32;
    fn c_mpi_world_rank(rank: *mut i32) -> i32;
    fn c_mpi_world_size(size: *mut i32) -> i32;
}

pub fn mpi_init() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize");
        }
    }
    Ok(())
}

pub fn mpi_init_threaded() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init_threaded();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize (threaded)");
        }
        if status != C_MPI_ERROR_INIT_THREADED {
            return Err("MPI failed to initialize a multithreaded setup");
        }
    }
    Ok(())
}

pub fn mpi_finalize() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_finalize();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to finalize");
        }
    }
    Ok(())
}

pub fn mpi_initialized() -> Result<bool, StrError> {
    unsafe {
        let mut flag: i32 = 0;
        let status = c_mpi_initialized(&mut flag);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot get initialized status");
        }
        Ok(if flag == 1 { true } else { false })
    }
}

pub fn mpi_world_rank() -> Result<usize, StrError> {
    unsafe {
        let mut rank: i32 = 0;
        let status = c_mpi_world_rank(&mut rank);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot failed to get world rank");
        }
        Ok(rank as usize)
    }
}

pub fn mpi_world_size() -> Result<usize, StrError> {
    unsafe {
        let mut size: i32 = 0;
        let status = c_mpi_world_size(&mut size);
        if status != C_MPI_SUCCESS {
            return Err("MPI cannot failed to get world size");
        }
        Ok(size as usize)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn essential_functions_work() {
        assert!(!mpi_initialized().unwrap());
        mpi_init().unwrap();
        assert!(mpi_initialized().unwrap());
        assert_eq!(mpi_world_rank().unwrap(), 0);
        assert_eq!(mpi_world_size().unwrap(), 1);
        mpi_finalize().unwrap();
    }
}
