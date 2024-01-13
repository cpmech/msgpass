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

    // broadcast -----------------------------------------------------------------------------

    fn comm_broadcast_i32(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut i32) -> i32;
    fn comm_broadcast_i64(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut i64) -> i32;
    fn comm_broadcast_u32(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut u32) -> i32;
    fn comm_broadcast_u64(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut u64) -> i32;
    fn comm_broadcast_f32(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut f32) -> i32;
    fn comm_broadcast_f64(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut f64) -> i32;
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
        if status == C_MPI_ERROR_INIT_THREADED {
            return Err("MPI failed to initialize a multithreaded setup");
        }
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize (threaded)");
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

    //  broadcast --------------------------------------------------------------------------------------

    pub fn broadcast_i32(&mut self, sender: usize, x: &mut [i32]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
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

    pub fn broadcast_i64(&mut self, sender: usize, x: &mut [i64]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_i64(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i64 slice");
            }
        }
        Ok(())
    }

    pub fn broadcast_u32(&mut self, sender: usize, x: &mut [u32]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_u32(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u32 slice");
            }
        }
        Ok(())
    }

    pub fn broadcast_u64(&mut self, sender: usize, x: &mut [u64]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_u64(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u64 slice");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_u64(self.ext_comm, c_sender, n, x.as_mut_ptr() as *mut u64);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u64 slice");
            }
        }
        Ok(())
    }

    #[cfg(target_pointer_width = "32")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_u32(self.ext_comm, c_sender, n, x.as_mut_ptr() as *mut u32);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u64 slice");
            }
        }
        Ok(())
    }

    pub fn broadcast_f32(&mut self, sender: usize, x: &mut [f32]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_f32(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f32 slice");
            }
        }
        Ok(())
    }

    pub fn broadcast_f64(&mut self, sender: usize, x: &mut [f64]) -> Result<(), StrError> {
        if x.len() < 1 {
            return Err("slice must have at least one component");
        }
        unsafe {
            let c_sender = to_i32(sender);
            let n = to_i32(x.len());
            let status = comm_broadcast_f64(self.ext_comm, c_sender, n, x.as_mut_ptr());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f64 slice");
            }
        }
        Ok(())
    }

    //  send -------------------------------------------------------------------------------------------
}
