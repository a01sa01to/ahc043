use std::time::{Duration, Instant};

fn main() {
    let start_time = Instant::now;

    // 5000ms まわす
    while start_time.elapsed() < Duration::from_millis(5000) {}
}
