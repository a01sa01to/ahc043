extern crate rand;
use proconio::{fastout, input};
use rand::{seq::SliceRandom, Rng};
use std::{cmp, collections, fmt, mem::swap, time::Instant};

const COST_STATION: usize = 5000;
const COST_RAIL: usize = 100;
const N: usize = 50;
const T: usize = 800;
const INF: usize = 10usize.pow(9);

const MASK_L: usize = 1;
const MASK_R: usize = 2;
const MASK_U: usize = 4;
const MASK_D: usize = 8;

const MANHATTAN_2_LIST: [(i32, i32); 13] = [
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GridState {
    Empty,
    Station(usize),
    Rail(RailType),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RailType {
    None = -1,
    Station = 0,
    LR = 1,
    UD = 2,
    LD = 3,
    LU = 4,
    RU = 5,
    RD = 6,
}
impl RailType {
    fn from_char(c: char) -> Self {
        match c {
            '#' => Self::Station,
            '-' => Self::LR,
            '|' => Self::UD,
            '\\' => Self::LD,
            'J' => Self::LU,
            'L' => Self::RU,
            '/' => Self::RD,
            _ => unreachable!(),
        }
    }
    fn to_char(&self) -> char {
        match self {
            RailType::None => unreachable!(),
            RailType::Station => '#',
            RailType::LR => '-',
            RailType::UD => '|',
            RailType::LD => '\\',
            RailType::LU => 'J',
            RailType::RU => 'L',
            RailType::RD => '/',
        }
    }
    fn from_mask(mask: usize) -> Self {
        assert!(mask.count_ones() == 2);
        if mask == (MASK_L | MASK_R) {
            return Self::LR;
        }
        if mask == (MASK_U | MASK_D) {
            return Self::UD;
        }
        if mask == (MASK_L | MASK_D) {
            return Self::LD;
        }
        if mask == (MASK_L | MASK_U) {
            return Self::LU;
        }
        if mask == (MASK_R | MASK_U) {
            return Self::RU;
        }
        if mask == (MASK_R | MASK_D) {
            return Self::RD;
        }
        unreachable!();
    }
}
impl fmt::Display for RailType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RailType::None => unreachable!(),
            RailType::Station => write!(f, "0"),
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
impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
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
    for (dx, dy) in MANHATTAN_2_LIST {
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
    Station::new(Point::new(x, y), num_new_users, num_known_users)
}

fn update_income(
    income: &mut usize,
    nconnected_peopleidx: &mut collections::HashSet<usize>,
    people: &Vec<Person>,
    grid_dsu: &mut ac_library::Dsu,
    grid_state: &Vec<Vec<GridState>>,
    pos2sta: &Vec<Vec<usize>>,
) {
    let mut done = collections::HashSet::new();
    for &i in nconnected_peopleidx.iter() {
        let p = &people[i];
        'outer: for (dx1, dy1) in MANHATTAN_2_LIST {
            if in_range(p.home.x as i32 + dx1, p.home.y as i32 + dy1) {
                let p1 = Point::new(
                    (p.home.x as i32 + dx1) as usize,
                    (p.home.y as i32 + dy1) as usize,
                );
                if grid_state[p1.x][p1.y] == GridState::Station(pos2sta[p1.x][p1.y]) {
                    for (dx2, dy2) in MANHATTAN_2_LIST {
                        if in_range(p.work.x as i32 + dx2, p.work.y as i32 + dy2) {
                            let p2 = Point::new(
                                (p.work.x as i32 + dx2) as usize,
                                (p.work.y as i32 + dy2) as usize,
                            );
                            if grid_dsu.same(p1.to_idx(), p2.to_idx())
                                && grid_state[p2.x][p2.y] == GridState::Station(pos2sta[p2.x][p2.y])
                            {
                                *income += p.dist();
                                done.insert(i);
                                break 'outer;
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

fn find_path(
    a: Point,
    to_sta: usize,
    next_pos: &Vec<Vec<Vec<Point>>>,
    grid_state: &Vec<Vec<GridState>>,
    pos2sta: &Vec<Vec<usize>>,
) -> (Vec<(RailType, Point)>, (usize, usize)) {
    let mut res = Vec::new();
    let mut num_sta = 0;
    let mut num_rail = 0;
    let mut now = a;

    while now != next_pos[to_sta][now.x][now.y] {
        let nxt = next_pos[to_sta][now.x][now.y];
        if now == a {
            // どうせ駅
            // Do nothing
        } else if grid_state[now.x][now.y] == GridState::Empty {
            // 線路はいったん RailType::None で埋める
            res.push((RailType::None, now));
            num_rail += 1;

            // Empty Empty なら線路で OK
            if grid_state[nxt.x][nxt.y] == GridState::Empty {
                // Do nothing
            }
            // もし次の部分が駅になる予定で、新しい方向から入ってくるなら駅を立てる
            else if pos2sta[nxt.x][nxt.y] != !0
                && grid_state[nxt.x][nxt.y] != GridState::Station(pos2sta[nxt.x][nxt.y])
            {
                res.push((RailType::Station, nxt));
                num_sta += 1;
            }
        } else {
            // 線路か駅
            let staidx = pos2sta[now.x][now.y];
            // 今後駅ができる かつ次が新しい方向なら駅をつくる
            if staidx != !0
                && grid_state[now.x][now.y] != GridState::Station(staidx)
                && grid_state[nxt.x][nxt.y] == GridState::Empty
            {
                res.push((RailType::Station, now));
                num_sta += 1;
            }
        }

        now = next_pos[to_sta][now.x][now.y];
    }

    (res, (num_sta, num_rail))
}

#[fastout]
fn output(ans: &Vec<((RailType, Point), (usize, usize))>) {
    for i in 0..ans.len() {
        let ((rail, pos), (money, income)) = ans[i];
        println!("# Turn: {}", i + 1);
        println!("# Money: {} Income: {}", money, income);
        if rail == RailType::None {
            println!("-1");
        } else {
            println!("{} {} {}", rail, pos.x, pos.y);
        }
    }
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
        let mut used = vec![vec![false; N]; N];

        let mut used_home = vec![false; m];
        let mut used_work = vec![false; m];

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
            for (dx, dy) in MANHATTAN_2_LIST {
                let nx = s.pos.x as i32 + dx;
                let ny = s.pos.y as i32 + dy;
                if !in_range(nx, ny) {
                    continue;
                }
                used[nx as usize][ny as usize] = true;
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
            for sta in pq.iter_mut() {
                *sta = get_station(
                    sta.pos.x,
                    sta.pos.y,
                    &people,
                    &grid_to_peopleidx,
                    &used_home,
                    &used_work,
                );
                for (dx, dy) in MANHATTAN_2_LIST {
                    let nx = sta.pos.x as i32 + dx;
                    let ny = sta.pos.y as i32 + dy;
                    if !in_range(nx, ny) {
                        continue;
                    }
                    if used[nx as usize][ny as usize] {
                        sta.num_new_users = 0;
                        sta.num_known_users = 0;
                    }
                }
            }
            pq.sort_by(|a, b| a.sum_users().cmp(&b.sum_users()));
            pq.reverse();
            while !pq.is_empty() && pq.last().unwrap().sum_users() == 0 {
                pq.pop();
            }
            pq.reverse();
        }
    }

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

        // 山登り
        let mut best_score = INF;
        let mut cnt = 0;
        while time.elapsed().as_millis() < 1000 {
            cnt += 1;
            if cnt % 300 == 0 {
                for _ in 0..100 {
                    let i = rng.gen_range(0..edges.len() - 1);
                    let (left, right) = edges.split_at_mut(i + 1);
                    swap(&mut left[i], &mut right[0]);
                }
            }

            for _ in 0..5 {
                let i = rng.gen_range(0..edges.len() - 1);
                let (left, right) = edges.split_at_mut(i + 1);
                swap(&mut left[i], &mut right[0]);
            }

            let mut cur = vec![vec!['.'; N]; N];
            let mut score = 0;
            let mut d = ac_library::Dsu::new(stations.len());
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
                let mut prv_pos = stations[*j].pos;
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

                    if cur[now_pos.x][now_pos.y] == '#' {
                        cur[now_pos.x][now_pos.y] = '#';
                    } else if prv_pos != now_pos {
                        // どの向きにつながるか
                        let mut mask = 0usize;
                        for &(q, msk) in &[
                            (now_pos.left(), MASK_L),
                            (now_pos.right(), MASK_R),
                            (now_pos.up(), MASK_U),
                            (now_pos.down(), MASK_D),
                        ] {
                            if q == prv_pos || q == next_pos {
                                mask |= msk;
                            }
                        }
                        cur[now_pos.x][now_pos.y] = RailType::from_mask(mask).to_char();
                    }

                    prv_pos = now_pos;
                    now_pos = next_pos;
                }
                cur[stations[*i].pos.x][stations[*i].pos.y] = '#';
                cur[stations[*j].pos.x][stations[*j].pos.y] = '#';

                d.merge(*i, *j);
                groups_cnt -= 1;
            }
            if score < best_score {
                best_score = score;
                target_grid = cur;
            }
        }
        eprintln!("cnt: {}", cnt);
        eprintln!("Score: {}", best_score);
    }

    {
        let mut sta_unused = Vec::new();
        for i in 0..stations.len() {
            if target_grid[stations[i].pos.x][stations[i].pos.y] != '#' {
                sta_unused.push(i);
            }
        }
        for &i in sta_unused.iter().rev() {
            stations.remove(i);
        }
        for (i, s) in stations.iter().enumerate() {
            assert_eq!(s.pos, stations[i].pos);
            eprintln!("Station {}: {} {}", i, s.pos.x, s.pos.y);
        }
    }
    let stations = stations;

    let people2sta = {
        let mut res = vec![(!0, !0); m];
        for (i, p) in people.iter().enumerate() {
            for (j, s) in stations.iter().enumerate() {
                for (dx, dy) in MANHATTAN_2_LIST {
                    let nx = s.pos.x as i32 + dx;
                    let ny = s.pos.y as i32 + dy;
                    let po = Point::new(nx as usize, ny as usize);
                    if po.in_range() && p.home == po {
                        res[i].0 = j;
                    }
                    if po.in_range() && p.work == po {
                        res[i].1 = j;
                    }
                }
            }
        }
        res
    };
    let sta2users = {
        let mut res = vec![Vec::new(); stations.len()];
        for i in 0..stations.len() {
            for j in 0..m {
                if people2sta[j].0 == !0 || people2sta[j].1 == !0 {
                    continue;
                }
                if people2sta[j].0 == i || people2sta[j].1 == i {
                    res[i].push(j);
                }
            }
        }
        res
    };

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
    let mut next_pos = vec![vec![vec![Point::new(!0, !0); N]; N]; stations.len()];
    let pos2sta = {
        let mut res = vec![vec![!0; N]; N];
        for (i, s) in stations.iter().enumerate() {
            res[s.pos.x][s.pos.y] = i;
        }
        res
    };
    {
        let graph = {
            let mut res = vec![vec![Vec::new(); N]; N];
            for i in 0..N {
                for j in 0..N {
                    if target_grid[i][j] == '.' {
                        continue;
                    }
                    let p = Point::new(i, j);
                    let mut cand = Vec::new();
                    if RailType::from_char(target_grid[i][j]) == RailType::Station {
                        cand = vec![p.left(), p.right(), p.up(), p.down()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::LR {
                        cand = vec![p.left(), p.right()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::UD {
                        cand = vec![p.up(), p.down()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::RU {
                        cand = vec![p.right(), p.up()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::LU {
                        cand = vec![p.left(), p.up()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::LD {
                        cand = vec![p.left(), p.down()];
                    }
                    if RailType::from_char(target_grid[i][j]) == RailType::RD {
                        cand = vec![p.right(), p.down()];
                    }
                    for &q in &cand {
                        if q.in_range() && target_grid[q.x][q.y] != '.' {
                            res[p.x][p.y].push(q);
                        }
                    }
                }
            }
            res
        };

        for (i, s) in stations.iter().enumerate() {
            dist[i][i] = 0;
            next_pos[i][s.pos.x][s.pos.y] = s.pos;

            let mut grid_dist = vec![vec![INF; N]; N];
            let mut que = collections::VecDeque::new();
            que.push_back(s.pos);
            grid_dist[s.pos.x][s.pos.y] = 0;
            while !que.is_empty() {
                let p = que.pop_front().unwrap();
                assert!(p.in_range());
                assert!(next_pos[i][p.x][p.y].in_range());
                for &q in &graph[p.x][p.y] {
                    assert!(q.in_range());
                    assert_ne!(target_grid[q.x][q.y], '.');
                    if grid_dist[q.x][q.y] == INF {
                        grid_dist[q.x][q.y] = grid_dist[p.x][p.y] + 1;
                        // 木になってるので次の位置は一意に定まる
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

    for i in 0..stations.len() {
        for j in 0..stations.len() {
            assert_ne!(
                dist[i][j], INF,
                "Sta{}{} -> Sta{}{}",
                i, stations[i].pos, j, stations[j].pos
            );
        }
    }

    for i in 0..N {
        for j in 0..N {
            if target_grid[i][j] != '.' {
                for s in 0..stations.len() {
                    assert_ne!(next_pos[s][i][j], Point::new(!0, !0), "{} {} {}", i, j, s);
                }
            }
        }
    }

    // 答えを出すパート
    let mut turn = 0;
    let mut income = 0;
    let mut nconnected_peopleidx = collections::HashSet::new();
    let mut build_todo = collections::VecDeque::new();
    let mut grid_dsu = ac_library::Dsu::new(N * N);
    let mut grid_state = vec![vec![GridState::Empty; N]; N];
    let mut dsu4stapair: Vec<Vec<ac_library::Dsu>> = (0..stations.len())
        .map(|_| {
            (0..stations.len())
                .map(|_| ac_library::Dsu::new(stations.len()))
                .collect()
        })
        .collect();

    // (output, ターン終了時の (money, income))
    let mut ans = vec![((RailType::None, Point::new(!0, !0)), (0, 0)); T];

    for i in 0..m {
        if people2sta[i].0 != !0 && people2sta[i].1 != !0 {
            nconnected_peopleidx.insert(i);
        }
    }
    for i in 0..stations.len() {
        for j in i + 1..stations.len() {
            dsu4stapair[i][j].merge(i, j);
        }
    }

    while turn < T {
        turn += 1;

        if !build_todo.is_empty() {
            let &(r, i, j) = build_todo.front().unwrap();
            let p = Point::new(i, j);
            assert!(p.in_range());
            assert_ne!(r, RailType::None);

            if r == RailType::Station && k >= COST_STATION {
                build_todo.pop_front();
                let staidx = pos2sta[i][j];
                assert_ne!(staidx, !0);
                grid_state[i][j] = GridState::Station(staidx);

                for &q in &[p.left(), p.right(), p.up(), p.down()] {
                    if q.in_range() && grid_state[q.x][q.y] != GridState::Empty {
                        grid_dsu.merge(p.to_idx(), q.to_idx());
                    }
                }

                k -= COST_STATION;
            } else if r != RailType::Station && k >= COST_RAIL {
                build_todo.pop_front();
                grid_state[i][j] = GridState::Rail(r);

                if r == RailType::LD || r == RailType::LR || r == RailType::LU {
                    assert!(p.left().in_range());
                    if grid_state[p.left().x][p.left().y] != GridState::Empty {
                        grid_dsu.merge(p.to_idx(), p.left().to_idx());
                    }
                }
                if r == RailType::RD || r == RailType::LR || r == RailType::RU {
                    assert!(p.right().in_range());
                    if grid_state[p.right().x][p.right().y] != GridState::Empty {
                        grid_dsu.merge(p.to_idx(), p.right().to_idx());
                    }
                }
                if r == RailType::LU || r == RailType::RU || r == RailType::UD {
                    assert!(p.up().in_range());
                    if grid_state[p.up().x][p.up().y] != GridState::Empty {
                        grid_dsu.merge(p.to_idx(), p.up().to_idx());
                    }
                }
                if r == RailType::LD || r == RailType::RD || r == RailType::UD {
                    assert!(p.down().in_range());
                    if grid_state[p.down().x][p.down().y] != GridState::Empty {
                        grid_dsu.merge(p.to_idx(), p.down().to_idx());
                    }
                }

                k -= COST_RAIL;
            } else {
                k += income;
                ans[turn - 1] = ((RailType::None, Point::new(!0, !0)), (k, income));
                continue;
            }

            update_income(
                &mut income,
                &mut nconnected_peopleidx,
                &people,
                &mut grid_dsu,
                &grid_state,
                &pos2sta,
            );

            k += income;

            ans[turn - 1] = ((r, p), (k, income));
            continue;
        }

        let mut best = (0, !0, !0);
        for i in 0..stations.len() {
            for j in i + 1..stations.len() {
                let mut profit = 0;

                let (cost, need_build_turn) = {
                    let (_, (addsta, addpath)) =
                        find_path(stations[i].pos, j, &next_pos, &grid_state, &pos2sta);

                    let mut nexista = 0;
                    if grid_state[stations[i].pos.x][stations[i].pos.y] != GridState::Station(i) {
                        nexista += 1;
                    }
                    if grid_state[stations[j].pos.x][stations[j].pos.y] != GridState::Station(j) {
                        nexista += 1;
                    }

                    (
                        ((addsta + nexista) * COST_STATION + addpath * COST_RAIL) as i32,
                        (addsta + addpath) as i32,
                    )
                };
                let need_wait_turn = if income == 0 {
                    0
                } else {
                    (cost as i32 - k as i32 + income as i32 - 1).max(0) / income as i32
                };

                let mut checked = collections::HashSet::new();
                for &idx in sta2users[i].iter().chain(sta2users[j].iter()) {
                    if dsu4stapair[i][j].same(people2sta[idx].0, people2sta[idx].1)
                        && nconnected_peopleidx.contains(&idx)
                        && !checked.contains(&idx)
                    {
                        let p = &people[idx];
                        profit += p.dist() as i32;
                        checked.insert(idx);
                    }
                }
                let score =
                    profit * ((T - (turn - 1)) as i32 - need_build_turn - need_wait_turn) - cost;
                // turn 1 なら制約付き
                if score > best.0 && (turn != 1 || k >= dist[i][j] * COST_RAIL + 2 * COST_STATION) {
                    best = (score, i, j);
                }
            }
        }
        // どうせ収益が減るならやめる
        if best.0 == 0 {
            for x in turn..=T {
                k += income;
                ans[x - 1] = ((RailType::None, Point::new(!0, !0)), (k, income));
            }
            break;
        }
        assert_ne!(best.1, !0);

        let (_, i, j) = best;
        turn -= 1; // 上のほうの処理に任せるため

        let (todos, _) = find_path(stations[i].pos, j, &next_pos, &grid_state, &pos2sta);
        let sipos = stations[i].pos;
        let sjpos = stations[j].pos;
        let mut sta_todo = Vec::new();
        if grid_state[sipos.x][sipos.y] != GridState::Station(i) {
            build_todo.push_back((RailType::Station, sipos.x, sipos.y));
            sta_todo.push(i);
        }
        for (r, p) in todos {
            if r == RailType::Station {
                build_todo.push_back((RailType::Station, p.x, p.y));
                sta_todo.push(pos2sta[p.x][p.y]);
                continue;
            }

            assert_eq!(r, RailType::None);

            if target_grid[p.x][p.y] == '#' {
                let mut mask = 0usize;
                for &(q, msk) in &[
                    (p.left(), MASK_L),
                    (p.right(), MASK_R),
                    (p.up(), MASK_U),
                    (p.down(), MASK_D),
                ] {
                    if q == next_pos[i][p.x][p.y] || q == next_pos[j][p.x][p.y] {
                        mask |= msk;
                    }
                }
                build_todo.push_back((RailType::from_mask(mask), p.x, p.y));
            } else {
                build_todo.push_back((RailType::from_char(target_grid[p.x][p.y]), p.x, p.y));
            }
        }
        if grid_state[sjpos.x][sjpos.y] != GridState::Station(j) {
            build_todo.push_back((RailType::Station, sjpos.x, sjpos.y));
            sta_todo.push(j);
        }
        for ii in 0..stations.len() {
            for ji in ii + 1..stations.len() {
                dsu4stapair[ii][ji].merge(i, j);
                if sta_todo.len() > 1 {
                    for idx in 0..sta_todo.len() - 1 {
                        dsu4stapair[ii][ji].merge(sta_todo[idx], sta_todo[idx + 1]);
                    }
                }
            }
        }
    }

    // 最終的に待ったほうがいいなら Revert
    for t in (0..T).rev() {
        if ans[T - 1].1 .0 < ans[t].1 .0 + ans[t].1 .1 * (T - t) {
            for x in t + 1..T {
                ans[x] = (
                    (RailType::None, Point::new(!0, !0)),
                    (ans[t].1 .0 + ans[t].1 .1 * (x - t), ans[t].1 .1),
                );
            }
        }
    }

    output(&ans);
}
