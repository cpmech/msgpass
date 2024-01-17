use std::env;

fn main() {
    let use_mpich = match env::var("MSGPASS_USE_MPICH") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };
    if use_mpich {
        cc::Build::new()
            .file("c_code/interface_mpi.c") // file
            .include("/usr/include/x86_64-linux-gnu/mpich/") // include
            .compile("c_code_interface_mpi"); // compile
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");
        println!("cargo:rustc-link-lib=dylib=mpich");
    } else {
        cc::Build::new()
            .file("c_code/interface_mpi.c") // file
            .include("/usr/lib/x86_64-linux-gnu/openmpi/include/") // include
            .compile("c_code_interface_mpi"); // compile
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openmpi");
        println!("cargo:rustc-link-lib=dylib=mpi");
    }
    println!("cargo:rerun-if-changed=c_code/constants.h");
    println!("cargo:rerun-if-changed=c_code/interface_mpi.c");
}
