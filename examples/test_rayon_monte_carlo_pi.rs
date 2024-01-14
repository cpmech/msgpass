use msgpass::*;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

// https://zsiciarz.github.io/24daysofrust/book/vol2/day3.html
// https://rust-random.github.io/book/guide-parallel.html

static SAMPLES: usize = 1_000;

fn monte_carlo_points_in_circle() -> usize {
    let range = Uniform::new(-1.0f64, 1.0);
    (0..SAMPLES)
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, _| {
                let x = range.sample(rng);
                let y = range.sample(rng);
                if x * x + y * y <= 1.0 {
                    1
                } else {
                    0
                }
            },
        )
        .reduce(|| 0usize, |a, b| a + b)
}

fn main() -> Result<(), StrError> {
    mpi_init_thread(MpiThread::Serialized)?;

    let rank = mpi_world_rank()?;
    let size = mpi_world_size()?;
    let mut comm = Communicator::new()?;

    let in_circle = monte_carlo_points_in_circle();
    println!("{}: π ≈ {}", rank, 4.0 * (in_circle as f64) / (SAMPLES as f64));

    let my_in_circle = [in_circle];
    let mut in_circle_sum = [0_usize];
    comm.reduce_usize(0, &mut in_circle_sum, &my_in_circle, MpiOp::Sum)?;

    if rank == 0 {
        let total = SAMPLES * size;
        let in_circ = in_circle_sum[0];
        println!("FINAL: π ≈ {}", 4.0 * (in_circ as f64) / (total as f64));
    }

    mpi_finalize()
}
