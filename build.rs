// validate intel source command --------------------------------------------------------------------------

#[cfg(feature = "intel_mkl")]
use std::env;

#[cfg(feature = "intel_mkl")]
fn validate_intel_setvars_completed() {
    let intel_setvars_completed = match env::var("SETVARS_COMPLETED") {
        Ok(v) => v == "1",
        Err(_) => false,
    };
    if !intel_setvars_completed {
        panic!("\n\nBUILD ERROR: Intel setvars.sh need to be sourced first.\nYou must execute the following command (just once):\nsource /opt/intel/oneapi/setvars.sh\n\n")
    }
}

#[cfg(not(feature = "intel_mkl"))]
fn validate_intel_setvars_completed() {}

// information --------------------------------------------------------------------------------------------// (default) Returns the directories and libraries

// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(not(feature = "intel"))]
#[cfg(not(feature = "mpich"))]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/include/", //
        ],
        // lib_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/", //
            "/usr/local/opt/open-mpi/lib/",       // macOS
        ],
        // libs
        vec![
            "mpi", //
        ],
    )
}

// (intel) Returns the directories and libraries
// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(feature = "intel")]
#[cfg(not(feature = "mpich"))]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/opt/intel/oneapi/mpi/latest/include/", //
        ],
        // lib_dirs
        vec![
            "/opt/intel/oneapi/mpi/latest/lib/", //
        ],
        // libs
        vec![
            "mpi", //
        ],
    )
}

// (mpich) Returns the directories and libraries
// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(feature = "mpich")]
#[cfg(not(feature = "intel"))]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/usr/include/x86_64-linux-gnu/mpich/", //
        ],
        // lib_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/", //
        ],
        // libs
        vec![
            "mpich", //
        ],
    )
}

// main ---------------------------------------------------------------------------------------------------

fn main() {
    // validate intel setvars
    validate_intel_setvars_completed();

    // information
    let (inc_dirs, lib_dirs, libs) = get_information();

    // compile the code
    cc::Build::new().file("c_code/interface_mpi.c").includes(&inc_dirs).compile("c_code_interface_mpi");

    // libraries
    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", *d);
    }
    for l in &libs {
        println!("cargo:rustc-link-lib=dylib={}", *l);
    }

    // watch changes
    println!("cargo:rerun-if-changed=c_code/constants.h");
    println!("cargo:rerun-if-changed=c_code/interface_mpi.c");
}
