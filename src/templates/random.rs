extern crate rand;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(0, 100);
}
