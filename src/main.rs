extern crate rand;
use proconio::input;
use rand::{seq::SliceRandom, Rng};
use std::{cmp, collections, fmt, mem::swap, time::Instant};

const COST_STATION: usize = 5000;
const COST_RAIL: usize = 100;
const N: usize = 50;
const T: usize = 800;
const INF: usize = 10usize.pow(9);

#[derive(Clone, Copy, PartialEq, Eq)]
enum RailType {
    LR = 1,
    UD = 2,
    LD = 3,
    LU = 4,
    RU = 5,
    RD = 6,
}
impl fmt::Display for RailType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RailType::LR => write!(f, "1"),
            RailType::UD => write!(f, "2"),
            RailType::LD => write!(f, "3"),
            RailType::LU => write!(f, "4"),
            RailType::RU => write!(f, "5"),
            RailType::RD => write!(f, "6"),
        }
    }
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
    fn to_idx(&self) -> usize {
        self.x * N + self.y
    }
}

fn manhattan_distance(p1: &Point, p2: &Point) -> u32 {
    ((p1.x as i32 - p2.x as i32).abs() + (p1.y as i32 - p2.y as i32).abs()) as u32
}
fn manhattan_distance_dxdy(dx: i32, dy: i32) -> u32 {
    (dx.abs() + dy.abs()) as u32
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
    fn dist(&self) -> usize {
        manhattan_distance(&self.home, &self.work) as usize
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
    let time = Instant::now();
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
                    if manhattan_distance_dxdy(dx, dy) <= 2 {
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
                    if manhattan_distance_dxdy(dx, dy) <= 2 {
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

    eprintln!("Time for finding station: {}ms", time.elapsed().as_millis());
    eprintln!("# of stations: {}", stations.len());

    // グラフを構築
    let mut target_grid = vec![vec!['.'; N]; N];
    {
        let mut edges = {
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

        let eq_range = {
            let mut res = Vec::new();
            let mut i = 0;
            while i < edges.len() {
                let mut j = i;
                while j < edges.len() && edges[j].0 == edges[i].0 {
                    j += 1;
                }
                res.push((i, j));
                i = j;
            }
            res
        };

        // 山登り
        let mut best_score = INF;
        while time.elapsed().as_millis() < 1500 {
            for (l, r) in &eq_range {
                edges[*l..*r].shuffle(&mut rng);
            }
            for _ in 0..10 {
                let i = rng.gen_range(0..edges.len() - 1);
                let (left, right) = edges.split_at_mut(i + 1);
                swap(&mut left[i], &mut right[0]);
            }

            let mut cnt = 0;
            let inner_time = Instant::now();
            while inner_time.elapsed().as_millis() < 100 {
                cnt += 1;
                let mut cur = vec![vec!['.'; N]; N];
                let mut score = 0;
                let mut d = ac_library::Dsu::new(stations.len());
                let idstr = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                let mut id = 0;
                let mut groups_cnt = stations.len();
                for (_dst, (i, j)) in &edges {
                    if groups_cnt == 1 {
                        break;
                    }
                    if d.same(*i, *j) {
                        continue;
                    }

                    // BFS
                    let mut grid_dist = vec![vec![INF; N]; N];
                    let mut que = collections::VecDeque::new();
                    que.push_back(stations[*i].pos);
                    grid_dist[stations[*i].pos.x][stations[*i].pos.y] = 0;
                    while !que.is_empty() {
                        let p = que.pop_front().unwrap();
                        if p == stations[*j].pos {
                            break;
                        }
                        for &q in &[p.left(), p.right(), p.up(), p.down()] {
                            if !q.in_range()
                                || grid_dist[q.x][q.y] != INF
                                || (cur[q.x][q.y] != '.' && cur[q.x][q.y] != '#')
                            {
                                continue;
                            }
                            grid_dist[q.x][q.y] = grid_dist[p.x][p.y] + 1;
                            que.push_back(q);
                        }
                    }
                    if grid_dist[stations[*j].pos.x][stations[*j].pos.y] == INF {
                        continue;
                    }
                    score += grid_dist[stations[*j].pos.x][stations[*j].pos.y];
                    let mut now_pos = stations[*j].pos;
                    while now_pos != stations[*i].pos {
                        let mut next_pos = now_pos;
                        let mut cand = vec![
                            now_pos.left(),
                            now_pos.right(),
                            now_pos.up(),
                            now_pos.down(),
                        ];
                        cand.shuffle(&mut rng);
                        for &q in &cand {
                            if !q.in_range() {
                                continue;
                            }
                            if grid_dist[q.x][q.y] + 1 == grid_dist[now_pos.x][now_pos.y] {
                                next_pos = q;
                            }
                        }
                        assert_ne!(next_pos, now_pos);
                        cur[now_pos.x][now_pos.y] = idstr.chars().nth(id).unwrap();
                        now_pos = next_pos;
                    }
                    cur[stations[*i].pos.x][stations[*i].pos.y] = '#';
                    cur[stations[*j].pos.x][stations[*j].pos.y] = '#';

                    d.merge(*i, *j);
                    groups_cnt -= 1;

                    id += 1;
                }
                if score < best_score {
                    best_score = score;
                    target_grid = cur;
                }
            }
            eprintln!("cnt: {}", cnt);
        }
        eprintln!("Score: {}", best_score);
    }

    eprintln!("Time for building graph: {}ms", time.elapsed().as_millis());
    eprintln!("Target Grid:");
    for i in 0..N {
        for j in 0..N {
            eprint!("{}", target_grid[i][j]);
        }
        eprintln!();
    }

    // target grid から dist と next_pos を作る
    let mut dist = vec![vec![INF; stations.len()]; stations.len()];
    let mut next_pos = vec![vec![vec![Point::new(!0, !0); stations.len()]; N]; N];
    {
        let mut pos2sta = vec![vec![!0; N]; N];
        for (i, s) in stations.iter().enumerate() {
            pos2sta[s.pos.x][s.pos.y] = i;
        }
        for (i, s) in stations.iter().enumerate() {
            dist[i][i] = 0;
            next_pos[i][s.pos.x][s.pos.y] = s.pos;

            let mut grid_dist = vec![vec![INF; N]; N];
            let mut que = collections::VecDeque::new();
            que.push_back(s.pos);
            while !que.is_empty() {
                let p = que.pop_front().unwrap();
                for &q in &[p.left(), p.right(), p.up(), p.down()] {
                    if !q.in_range() || target_grid[q.x][q.y] == '.' {
                        continue;
                    }
                    if grid_dist[q.x][q.y] == INF {
                        grid_dist[q.x][q.y] = grid_dist[p.x][p.y] + 1;
                        // 木になってるので、次の位置は一意に定まる
                        next_pos[i][q.x][q.y] = p;
                        que.push_back(q);
                        if target_grid[q.x][q.y] == '#' {
                            assert_ne!(pos2sta[q.x][q.y], !0);
                            let j = pos2sta[q.x][q.y];
                            dist[i][j] = grid_dist[q.x][q.y];
                        }
                    }
                }
            }
        }
    }

    // 答えを出すパート
    let mut turn = 0;
    let mut income = 0;
    let mut nconnected_peopleidx = collections::HashSet::new();
    let mut rail_todo = collections::VecDeque::new();
    let mut station_todo = collections::VecDeque::new();
    let mut grid_dsu = ac_library::Dsu::new(N * N);

    for i in 0..m {
        nconnected_peopleidx.insert(i);
    }

    fn update_income(
        income: &mut usize,
        nconnected_peopleidx: &mut collections::HashSet<usize>,
        people: &Vec<Person>,
        grid_dsu: &mut ac_library::Dsu,
    ) {
        let mut done = collections::HashSet::new();
        for &i in nconnected_peopleidx.iter() {
            if done.contains(&i) {
                continue;
            }
            let p: &Person = &people[i];
            for dx1 in -2i32..=2i32 {
                for dy1 in -2i32..=2i32 {
                    for dx2 in -2i32..=2i32 {
                        for dy2 in -2i32..=2i32 {
                            if manhattan_distance_dxdy(dx1, dy1) <= 2
                                && manhattan_distance_dxdy(dx2, dy2) <= 2
                                && in_range(p.home.x as i32 + dx1, p.home.y as i32 + dy1)
                                && in_range(p.work.x as i32 + dx2, p.work.y as i32 + dy2)
                            {
                                let p1 = Point::new(
                                    (p.home.x as i32 + dx1) as usize,
                                    (p.home.y as i32 + dy1) as usize,
                                );
                                let p2 = Point::new(
                                    (p.work.x as i32 + dx2) as usize,
                                    (p.work.y as i32 + dy2) as usize,
                                );
                                if grid_dsu.same(p1.to_idx(), p2.to_idx()) {
                                    *income += p.dist();
                                    done.insert(i);
                                }
                            }
                        }
                    }
                }
            }
        }
        for &i in &done {
            nconnected_peopleidx.remove(&i);
        }
    }

    while turn < T {
        turn += 1;

        if !station_todo.is_empty() && k >= COST_STATION {
            let i = station_todo.pop_front().unwrap();
            let s: &Station = &stations[i];
            println!("0 {} {}", s.pos.x, s.pos.y);

            for &q in &[s.pos.left(), s.pos.right(), s.pos.up(), s.pos.down()] {
                if !q.in_range() {
                    continue;
                }
                grid_dsu.merge(s.pos.to_idx(), q.to_idx());
            }
            update_income(
                &mut income,
                &mut nconnected_peopleidx,
                &people,
                &mut grid_dsu,
            );

            k -= COST_STATION;
            k += income;
            continue;
        }

        if !rail_todo.is_empty() && k >= COST_RAIL {
            let (t, i, j) = rail_todo.pop_front().unwrap();
            let p = Point::new(i, j);
            println!("{} {} {}", t, i, j);

            let mut cand = Vec::new();
            if t == RailType::LD || t == RailType::LR || t == RailType::LU {
                assert!(p.left().in_range());
                cand.push(p.left());
            }
            if t == RailType::RD || t == RailType::LR || t == RailType::RU {
                assert!(p.right().in_range());
                cand.push(p.right());
            }
            if t == RailType::LU || t == RailType::RU || t == RailType::UD {
                assert!(p.up().in_range());
                cand.push(p.up());
            }
            if t == RailType::LD || t == RailType::RD || t == RailType::UD {
                assert!(p.down().in_range());
                cand.push(p.down());
            }

            for &q in &cand {
                grid_dsu.merge(p.to_idx(), q.to_idx());
            }
            update_income(
                &mut income,
                &mut nconnected_peopleidx,
                &people,
                &mut grid_dsu,
            );

            k -= COST_RAIL;
            k += income;
            continue;
        }

        if !rail_todo.is_empty() || !station_todo.is_empty() {
            println!("-1");
            k += income;
            continue;
        }

        // TODO: impl
        // 型付けのために push
        rail_todo.push_back((RailType::LR, 0usize, 1usize));
        station_todo.push_back(0usize);
    }
}
