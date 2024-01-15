use crate::constants::*;
use crate::conversion::to_i32;
use crate::enums::*;
use crate::StrError;
use std::ffi::c_void;

#[repr(C)]
struct ExtCommunicator {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn c_mpi_init() -> i32;
    fn c_mpi_init_thread(option_index: i32) -> i32;
    fn c_mpi_finalize() -> i32;
    fn c_mpi_initialized(flag: *mut i32) -> i32;
    fn c_mpi_world_rank(rank: *mut i32) -> i32;
    fn c_mpi_world_size(size: *mut i32) -> i32;
    fn comm_drop(comm: *mut ExtCommunicator);
    fn comm_new() -> *mut ExtCommunicator;
    fn comm_new_subset(n_rank: i32, ranks: *const i32) -> *mut ExtCommunicator;
    fn comm_abort(comm: *mut ExtCommunicator, error_code: i32) -> i32;
    fn comm_barrier(comm: *mut ExtCommunicator) -> i32;
    fn comm_rank(comm: *mut ExtCommunicator, rank: *mut i32) -> i32;
    fn comm_size(comm: *mut ExtCommunicator, size: *mut i32) -> i32;
    fn comm_broadcast(comm: *mut ExtCommunicator, sender: i32, n: i32, x: *mut c_void, type_index: i32) -> i32;
    fn comm_reduce(comm: *mut ExtCommunicator, root: i32, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32, op_index: i32) -> i32;
    fn comm_allreduce(comm: *mut ExtCommunicator, n: i32, dest: *mut c_void, orig: *const c_void, type_index: i32, op_index: i32) -> i32;
    fn comm_send(comm: *mut ExtCommunicator, n: i32, data: *const c_void, type_index: i32, to_rank: i32, tag: i32) -> i32;
    fn comm_receive(comm: *mut ExtCommunicator, n: i32, data: *mut c_void, type_index: i32, from_rank: i32, tag: i32) -> i32;
    fn comm_get_receive_status(comm: *mut ExtCommunicator, source: *mut i32, tag: *mut i32, error: *mut i32);
}

/// Initializes the MPI execution environment
pub fn mpi_init() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize");
        }
    }
    Ok(())
}

/// Initializes the MPI execution environment (with thread options)
///
/// See [MpiThread]
pub fn mpi_init_thread(option: MpiThread) -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_init_thread(option.n());
        if status == C_MPI_ERROR_INIT_THREADED {
            return Err("MPI failed to match the required thread option");
        }
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to initialize (threaded)");
        }
    }
    Ok(())
}

/// Terminates the MPI execution environment
pub fn mpi_finalize() -> Result<(), StrError> {
    unsafe {
        let status = c_mpi_finalize();
        if status != C_MPI_SUCCESS {
            return Err("MPI failed to finalize");
        }
    }
    Ok(())
}

/// Checks whether MPI has been initialized or not
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

/// Determines the rank of the calling process in the MPI_COMM_WORLD communicator
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

/// Returns the size of the group associated with the MPI_COMM_WORLD communicator
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

/// Implements the Rust communicator (wrapping the C data)
///
/// This struct holds a pointer to the C communicator, which stores the communicator (MPI_Comm),
/// the group (MPI_Group), and the status from recv calls (MPI_Status).
pub struct Communicator {
    handle: *mut ExtCommunicator,
}

impl Drop for Communicator {
    /// Deallocates the C memory
    fn drop(&mut self) {
        unsafe {
            comm_drop(self.handle);
        }
    }
}

impl Communicator {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let ext_comm = comm_new();
            if ext_comm.is_null() {
                return Err("MPI failed to return the world communicator");
            }
            Ok(Communicator { handle: ext_comm })
        }
    }

    /// Allocates a new instance using a subset of processors
    pub fn new_subset(ranks: &[usize]) -> Result<Self, StrError> {
        unsafe {
            let n = to_i32(ranks.len());
            let c_ranks: Vec<i32> = ranks.iter().map(|r| *r as i32).collect();
            let ext_comm = comm_new_subset(n, c_ranks.as_ptr());
            if ext_comm.is_null() {
                return Err("MPI failed to create subset communicator");
            }
            Ok(Communicator { handle: ext_comm })
        }
    }

    /// Terminates the MPI execution environment
    pub fn abort(&mut self, error_code: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_abort(self.handle, error_code);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to abort");
            }
        }
        Ok(())
    }

    /// Synchronizes the MPI processes
    pub fn barrier(&mut self) -> Result<(), StrError> {
        unsafe {
            let status = comm_barrier(self.handle);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to synchronize (barrier)");
            }
        }
        Ok(())
    }

    /// Determines the rank of the calling process in the communicator
    pub fn rank(&mut self) -> Result<usize, StrError> {
        let mut rank: i32 = 0;
        unsafe {
            let status = comm_rank(self.handle, &mut rank);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to get the rank");
            }
        }
        Ok(rank as usize)
    }

    /// Returns the size of the group associated with a communicator
    pub fn size(&mut self) -> Result<usize, StrError> {
        let mut size: i32 = 0;
        unsafe {
            let status = comm_size(self.handle, &mut size);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to get the size");
            }
        }
        Ok(size as usize)
    }

    //  broadcast --------------------------------------------------------------------------------------

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_i32(&mut self, sender: usize, x: &mut [i32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::I32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_i64(&mut self, sender: usize, x: &mut [i64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::I64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast i64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_u32(&mut self, sender: usize, x: &mut [u32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_u64(&mut self, sender: usize, x: &mut [u64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast u64 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    #[cfg(target_pointer_width = "64")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast usize array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    #[cfg(target_pointer_width = "32")]
    pub fn broadcast_usize(&mut self, sender: usize, x: &mut [usize]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.ext_comm, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::U32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast usize array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_f32(&mut self, sender: usize, x: &mut [f32]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::F32.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f32 array");
            }
        }
        Ok(())
    }

    /// Broadcasts a message from sender to all other processes in the group
    pub fn broadcast_f64(&mut self, sender: usize, x: &mut [f64]) -> Result<(), StrError> {
        unsafe {
            let status = comm_broadcast(self.handle, to_i32(sender), to_i32(x.len()), x.as_mut_ptr() as *mut c_void, MpiType::F64.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to broadcast f64 array");
            }
        }
        Ok(())
    }

    // reduce -----------------------------------------------------------------------------------------

    /// Reduces values on all processes within a group
    pub fn reduce_i32(&mut self, root: usize, dest: &mut [i32], orig: &[i32], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce i32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_i64(&mut self, root: usize, dest: &mut [i64], orig: &[i64], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce i64 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_u32(&mut self, root: usize, dest: &mut [u32], orig: &[u32], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce u32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_u64(&mut self, root: usize, dest: &mut [u64], orig: &[u64], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce u64 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    #[cfg(target_pointer_width = "64")]
    pub fn reduce_usize(&mut self, root: usize, dest: &mut [usize], orig: &[usize], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce usize array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    #[cfg(target_pointer_width = "32")]
    pub fn reduce_usize(&mut self, root: usize, dest: &mut [usize], orig: &[usize], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce usize array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_f32(&mut self, root: usize, dest: &mut [f32], orig: &[f32], op: MpiOpx) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce f32 array");
            }
        }
        Ok(())
    }

    /// Reduces values on all processes within a group
    pub fn reduce_f64(&mut self, root: usize, dest: &mut [f64], orig: &[f64], op: MpiOpx) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_reduce(self.handle, to_i32(root), to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to reduce f64 array");
            }
        }
        Ok(())
    }

    // allreduce -----------------------------------------------------------------------------------------

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_i32(&mut self, dest: &mut [i32], orig: &[i32], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce i32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_i64(&mut self, dest: &mut [i64], orig: &[i64], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::I64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce i64 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_u32(&mut self, dest: &mut [u32], orig: &[u32], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce u32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_u64(&mut self, dest: &mut [u64], orig: &[u64], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce u64 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    #[cfg(target_pointer_width = "64")]
    pub fn allreduce_usize(&mut self, dest: &mut [usize], orig: &[usize], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce usize array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    #[cfg(target_pointer_width = "32")]
    pub fn allreduce_usize(&mut self, dest: &mut [usize], orig: &[usize], op: MpiOp) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::U32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce usize array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_f32(&mut self, dest: &mut [f32], orig: &[f32], op: MpiOpx) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F32.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce f32 array");
            }
        }
        Ok(())
    }

    /// Combines values from all processes and distributes the result back to all processes
    pub fn allreduce_f64(&mut self, dest: &mut [f64], orig: &[f64], op: MpiOpx) -> Result<(), StrError> {
        if dest.len() != orig.len() {
            return Err("arrays must have the same size");
        }
        unsafe {
            let status = comm_allreduce(self.handle, to_i32(orig.len()), dest.as_mut_ptr() as *mut c_void, orig.as_ptr() as *const c_void, MpiType::F64.n(), op.n());
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to (all) reduce f64 array");
            }
        }
        Ok(())
    }

    // getters -------------------------------------------------------------------------------------------

    /// Returns the status of the last receive call
    pub fn get_receive_status(&mut self) -> (i32, i32, i32) {
        let mut source = 0;
        let mut tag = 0;
        let mut error = 0;
        unsafe {
            comm_get_receive_status(self.handle, &mut source, &mut tag, &mut error);
        }
        (source, tag, error)
    }

    // send ----------------------------------------------------------------------------------------------

    /// Performs a standard-mode blocking send
    pub fn send_i32(&mut self, data: &[i32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::I32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send i32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_i64(&mut self, data: &[i64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::I64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send i64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_u32(&mut self, data: &[u32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send u32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_u64(&mut self, data: &[u64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send u64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    #[cfg(target_pointer_width = "64")]
    pub fn send_usize(&mut self, data: &[usize], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    #[cfg(target_pointer_width = "32")]
    pub fn send_usize(&mut self, data: &[usize], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::U32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_f32(&mut self, data: &[f32], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::F32.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send f32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking send
    pub fn send_f64(&mut self, data: &[f64], to_rank: usize, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_send(self.handle, to_i32(data.len()), data.as_ptr() as *const c_void, MpiType::F64.n(), to_i32(to_rank), tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to send f64 array");
            }
        }
        Ok(())
    }

    // receive -------------------------------------------------------------------------------------------

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_i32(&mut self, data: &mut [i32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::I32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive i32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_i64(&mut self, data: &mut [i64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::I64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive i64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_u32(&mut self, data: &mut [u32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive u32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_u64(&mut self, data: &mut [u64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive u64 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    #[cfg(target_pointer_width = "64")]
    pub fn receive_usize(&mut self, data: &mut [usize], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    #[cfg(target_pointer_width = "32")]
    pub fn receive_usize(&mut self, data: &mut [usize], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::U32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive usize array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_f32(&mut self, data: &mut [f32], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::F32.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive f32 array");
            }
        }
        Ok(())
    }

    /// Performs a standard-mode blocking receive
    ///
    /// `data` -- Buffer to store the received data
    /// `from_rank` -- Rank from where the data was sent (a negative value corresponds to MPI_ANY_SOURCE)
    /// `tag` -- Tag of the message (a negative value corresponds to MPI_ANY_TAG)
    pub fn receive_f64(&mut self, data: &mut [f64], from_rank: i32, tag: i32) -> Result<(), StrError> {
        unsafe {
            let status = comm_receive(self.handle, to_i32(data.len()), data.as_mut_ptr() as *mut c_void, MpiType::F64.n(), from_rank, tag);
            if status != C_MPI_SUCCESS {
                return Err("MPI failed to receive f64 array");
            }
        }
        Ok(())
    }
}
