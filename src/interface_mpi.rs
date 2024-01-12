extern "C" {
    fn mpi_hello_world();
}

pub fn hello_world() {
    unsafe {
        mpi_hello_world();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::hello_world;

    #[test]
    fn hello_world_works() {
        hello_world();
    }
}
