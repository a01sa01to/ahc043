use proconio::input;
use std::str;
use std::time::{Duration, Instant};

const COST_STATION: u32 = 5000;
const COST_RAIL: u32 = 100;

enum RailType {
    LR = 1,
    UD = 2,
    LD = 3,
    LU = 4,
    RU = 5,
    RD = 6,
}

#[derive(Clone, Copy)]
struct Point {
    x: usize,
    y: usize,
}
impl Point {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

fn manhattan_distance(p1: &Point, p2: &Point) -> u32 {
    ((p1.x as i32 - p2.x as i32).abs() + (p1.y as i32 - p2.y as i32).abs()) as u32
}

#[derive(Clone, Copy)]
struct Person {
    home: Point,
    work: Point,
}
impl Person {
    fn new(home: Point, work: Point) -> Self {
        Self { home, work }
    }
    fn dist(&self) -> u32 {
        manhattan_distance(&self.home, &self.work)
    }
}

struct Station {}

fn main() {
    let start_time = Instant::now;

    input! {
        n: usize,
        m: usize,
        k: usize,
        t: usize,
    };
    let mut people = Vec::new();
    for _ in 0..m {
        input! {
            x1: usize,
            y1: usize,
            x2: usize,
            y2: usize,
        };
        people.push(Person::new(Point::new(x1, y1), Point::new(x2, y2)));
    }
    let people = people;
}
