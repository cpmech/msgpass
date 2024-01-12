fn main() {
    cc::Build::new()
        .file("c_code/interface_mpi.c")
        .include("/usr/lib/x86_64-linux-gnu/openmpi/include/")
        .compile("c_code_interface_mpi");
}
