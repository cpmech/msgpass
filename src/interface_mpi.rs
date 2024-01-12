use crate::constants::*;
use crate::conversion::to_i32;
use crate::StrError;

#[repr(C)]
pub(crate) struct ExtCommunicator {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn c_mpi_init() -> i32;
    fn c_mpi_init_threaded() -> i32;
    fn c_mpi_finalize() -> i32;
    fn c_mpi_initialized(flag: *mut i32) -> i32;
    fn c_mpi_world_rank(rank: *mut i32) -> i32;
    fn c_mpi_world_size(size: *mut i32) -> i32;
    fn comm_drop(comm: *mut ExtCommunicator);
    fn comm_new() -> *mut ExtCommunicator;
    fn comm_new_subset(n_rank: i32, ranks: *const i32) -> *mut ExtCommunicator;
    fn comm_abort(comm: *mut ExtCommunicator) -> i32;
    fn comm_broadcast_i32(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut i32) -> i32;

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

pub struct Communicator {
    ext_comm: *mut ExtCommunicator,
}

impl Drop for Communicator {
    fn drop(&mut self) {
        unsafe {
            comm_drop(self.ext_comm);
        }
    }
}

impl Communicator {
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let ext_comm = comm_new();
            if ext_comm.is_null() {
                return Err("MPI failed to return the world communicator");
            }
            Ok(Communicator { ext_comm })
        }
    }

    pub fn new_subset(ranks: &[usize]) -> Result<Self, StrError> {
        unsafe {
            let n = to_i32(ranks.len());
            let c_ranks: Vec<i32> = ranks.iter().map(|r| *r as i32).collect();
            let ext_comm = comm_new_subset(n, c_ranks.as_ptr());
            if ext_comm.is_null() {
                return Err("MPI failed to create subset communicator");
            }
            Ok(Communicator { ext_comm })
        }
    }

    pub fn abort(&mut self) -> Result<(), StrError> {
        unsafe {
            let status = comm_abort(self.ext_comm);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to abort");
            }
        }
        Ok(())
    }

    pub fn broadcast_i32(&mut self, sender: usize, x: &mut [i32]) -> Result<(), StrError> {
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_i32(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i32 slice");
            }
        }
        Ok(())
    }
}
