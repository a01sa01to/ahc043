extern crate rand;
use proconio::input;
use rand::Rng;
use std::{cmp, collections, time::Instant};

const COST_STATION: u32 = 5000;
const COST_RAIL: u32 = 100;
const N: usize = 50;
const T: usize = 800;
const INF: u32 = 10u32.pow(9);

enum RailType {
    LR = 1,
    UD = 2,
    LD = 3,
    LU = 4,
    RU = 5,
    RD = 6,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Point {
    x: usize,
    y: usize,
}
impl Point {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
    fn left(&self) -> Self {
        if self.y == 0 {
            return Self::new(self.x, N);
        }
        Self::new(self.x, self.y - 1)
    }
    fn right(&self) -> Self {
        Self::new(self.x, self.y + 1)
    }
    fn up(&self) -> Self {
        if self.x == 0 {
            return Self::new(N, self.y);
        }
        Self::new(self.x - 1, self.y)
    }
    fn down(&self) -> Self {
        Self::new(self.x + 1, self.y)
    }
    fn in_range(&self) -> bool {
        self.x < N && self.y < N
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Station {
    pos: Point,
    num_new_users: usize,
    num_known_users: usize,
}
impl Station {
    fn new(pos: Point, num_new_users: usize, num_known_users: usize) -> Self {
        Self {
            pos,
            num_new_users,
            num_known_users,
        }
    }
    fn sum_users(&self) -> usize {
        self.num_new_users + self.num_known_users
    }
}
impl cmp::PartialOrd for Station {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        if self.sum_users() != other.sum_users() {
            self.sum_users().partial_cmp(&other.sum_users())
        } else {
            self.num_known_users.partial_cmp(&other.num_known_users)
        }
    }
}
impl cmp::Ord for Station {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self.sum_users() != other.sum_users() {
            self.sum_users().cmp(&other.sum_users())
        } else {
            self.num_known_users.cmp(&other.num_known_users)
        }
    }
}

fn in_range(x: i32, y: i32) -> bool {
    x >= 0 && x < N as i32 && y >= 0 && y < N as i32
}

fn main() {
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();

    // Input
    input! {
        _n: usize,
        m: usize,
        mut k: usize,
        _t: usize,
    };
    assert_eq!(_n, N);
    assert_eq!(_t, T);
    let people = {
        let mut res = Vec::new();
        for _ in 0..m {
            input! {
                x1: usize,
                y1: usize,
                x2: usize,
                y2: usize,
            };
            res.push(Person::new(Point::new(x1, y1), Point::new(x2, y2)));
        }
        res
    };

    let grid_to_peopleidx = {
        let mut res = vec![vec![Vec::new(); N]; N];
        for i in 0..m {
            res[people[i].home.x][people[i].home.y].push(i);
            res[people[i].work.x][people[i].work.y].push(i);
        }
        res
    };

    // 駅の場所を決める
    let mut stations = Vec::new();
    {
        let mut used_home = vec![false; m];
        let mut used_work = vec![false; m];

        fn get_station(
            x: usize,
            y: usize,
            people: &Vec<Person>,
            grid_to_peopleidx: &Vec<Vec<Vec<usize>>>,
            used_home: &Vec<bool>,
            used_work: &Vec<bool>,
        ) -> Station {
            let mut num_new_users = 0;
            let mut num_known_users = 0;
            for dx in -2i32..=2i32 {
                for dy in -2i32..=2i32 {
                    if dx.abs() + dy.abs() <= 2 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if !in_range(nx, ny) {
                            continue;
                        }
                        let p = Point::new(nx as usize, ny as usize);
                        for &i in &grid_to_peopleidx[nx as usize][ny as usize] {
                            if used_home[i] && used_work[i] {
                                continue;
                            } else if used_home[i] && people[i].work == p {
                                num_known_users += 1;
                            } else if used_work[i] && people[i].home == p {
                                num_known_users += 1;
                            } else if !used_home[i] && !used_work[i] {
                                num_new_users += 1;
                            }
                        }
                    }
                }
            }
            Station::new(Point::new(x, y), num_new_users, num_known_users)
        }

        let mut pq = Vec::new();
        for x in 0..N {
            for y in 0..N {
                pq.push(get_station(
                    x,
                    y,
                    &people,
                    &grid_to_peopleidx,
                    &used_home,
                    &used_work,
                ));
            }
        }
        pq.sort_by(|a, b| a.sum_users().cmp(&b.sum_users()));
        while !pq.is_empty() {
            let s = pq.pop().unwrap();
            stations.push(s);
            for dx in -2i32..=2i32 {
                for dy in -2i32..=2i32 {
                    if dx.abs() + dy.abs() <= 2 {
                        let nx = s.pos.x as i32 + dx;
                        let ny = s.pos.y as i32 + dy;
                        if !in_range(nx, ny) {
                            continue;
                        }
                        let p = Point::new(nx as usize, ny as usize);
                        for &i in &grid_to_peopleidx[nx as usize][ny as usize] {
                            if people[i].home == p {
                                used_home[i] = true;
                            }
                            if people[i].work == p {
                                used_work[i] = true;
                            }
                        }
                    }
                }
            }
            for sta in pq.iter_mut() {
                *sta = get_station(
                    sta.pos.x,
                    sta.pos.y,
                    &people,
                    &grid_to_peopleidx,
                    &used_home,
                    &used_work,
                );
            }
            pq.sort_by(|a, b| a.sum_users().cmp(&b.sum_users()));
            pq.reverse();
            while !pq.is_empty() && pq.last().unwrap().sum_users() == 0 {
                pq.pop();
            }
            pq.reverse();
        }
    }
    let stations = stations;

    eprintln!(
        "Time for finding station: {}ms",
        start_time.elapsed().as_millis()
    );
    eprintln!("# of stations: {}", stations.len());

    // グラフを構築
    // MST なので next_pos は一意だがまあ楽だし...
    // let mut cost = 0;
    let mut graph = vec![Vec::new(); stations.len()];
    let mut target_grid = vec![vec!['.'; N]; N];
    // let mut dist = vec![vec![INF; stations.len()]; stations.len()];
    // let mut next_pos = vec![vec![!0; stations.len()]; stations.len()];
    {
        let edges = {
            let mut res = Vec::new();
            for i in 0..stations.len() {
                for j in i + 1..stations.len() {
                    let d = manhattan_distance(&stations[i].pos, &stations[j].pos);
                    res.push((d, (i, j)));
                }
            }
            res.sort();
            res
        };

        let mut d = ac_library::Dsu::new(stations.len());
        let mut edges_tobeused = Vec::new();
        let idstr = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut id = 0;
        for (dst, (i, j)) in edges {
            if d.same(i, j) {
                continue;
            }

            // BFS
            let mut grid_dist = vec![vec![INF; N]; N];
            let mut que = collections::VecDeque::new();
            que.push_back(stations[i].pos);
            grid_dist[stations[i].pos.x][stations[i].pos.y] = 0;
            while !que.is_empty() {
                let p = que.pop_front().unwrap();
                for &q in &[p.left(), p.right(), p.up(), p.down()] {
                    if !q.in_range()
                        || grid_dist[q.x][q.y] != INF
                        || (target_grid[q.x][q.y] != '.' && target_grid[q.x][q.y] != '#')
                    {
                        continue;
                    }
                    grid_dist[q.x][q.y] = grid_dist[p.x][p.y] + 1;
                    que.push_back(q);
                }
            }
            if grid_dist[stations[j].pos.x][stations[j].pos.y] == INF {
                continue;
            }
            let mut now_pos = stations[j].pos;
            while now_pos != stations[i].pos {
                let mut next_pos = now_pos;
                for &q in &[
                    now_pos.left(),
                    now_pos.right(),
                    now_pos.up(),
                    now_pos.down(),
                ] {
                    if !q.in_range() {
                        continue;
                    }
                    if grid_dist[q.x][q.y] + 1 == grid_dist[now_pos.x][now_pos.y] {
                        next_pos = q;
                        break;
                    }
                }
                assert_ne!(next_pos, now_pos);
                target_grid[now_pos.x][now_pos.y] = idstr.chars().nth(id).unwrap();
                now_pos = next_pos;
            }
            target_grid[stations[i].pos.x][stations[i].pos.y] = '#';
            target_grid[stations[j].pos.x][stations[j].pos.y] = '#';

            d.merge(i, j);
            graph[i].push(j);
            graph[j].push(i);
            edges_tobeused.push((dst, (i, j)));

            id += 1;
        }
        assert_eq!(edges_tobeused.len(), stations.len() - 1);
    }

    eprintln!(
        "Time for building graph: {}ms",
        start_time.elapsed().as_millis()
    );
    eprintln!("Target Grid:");
    for i in 0..N {
        for j in 0..N {
            eprint!("{}", target_grid[i][j]);
        }
        eprintln!();
    }

    for s in &stations {
        println!("0 {} {}", s.pos.x, s.pos.y);
    }
    for _ in stations.len()..T {
        println!("-1");
    }
}
