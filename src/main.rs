extern crate rand;
use proconio::{fastout, input};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{cmp::Reverse, collections, fmt, time};

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

#[derive(PartialEq)]
enum SolverType {
    Type1,
    Type2,
}

#[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
enum Direction {
    None,
    Left,
    Right,
    Up,
    Down,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GridState {
    Empty = -1,
    Station = 0,
    LR = 1,
    UD = 2,
    LD = 3,
    LU = 4,
    RU = 5,
    RD = 6,
}
impl GridState {
    fn to_char(&self) -> char {
        match self {
            GridState::Empty => unreachable!(),
            GridState::Station => '#',
            GridState::LR => '-',
            GridState::UD => '|',
            GridState::LD => '\\',
            GridState::LU => 'J',
            GridState::RU => 'L',
            GridState::RD => '/',
        }
    }
    fn output(&self) -> char {
        match self {
            GridState::Empty => unreachable!(),
            GridState::Station => '0',
            GridState::LR => '1',
            GridState::UD => '2',
            GridState::LD => '3',
            GridState::LU => '4',
            GridState::RU => '5',
            GridState::RD => '6',
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
    fn can_conn_left(&self) -> bool {
        match self {
            GridState::Empty => false,
            GridState::Station => true,
            GridState::LR => true,
            GridState::UD => false,
            GridState::LD => true,
            GridState::LU => true,
            GridState::RU => false,
            GridState::RD => false,
        }
    }
    fn can_conn_right(&self) -> bool {
        match self {
            GridState::Empty => false,
            GridState::Station => true,
            GridState::LR => true,
            GridState::UD => false,
            GridState::LD => false,
            GridState::LU => false,
            GridState::RU => true,
            GridState::RD => true,
        }
    }
    fn can_conn_up(&self) -> bool {
        match self {
            GridState::Empty => false,
            GridState::Station => true,
            GridState::LR => false,
            GridState::UD => true,
            GridState::LD => false,
            GridState::LU => true,
            GridState::RU => true,
            GridState::RD => false,
        }
    }
    fn can_conn_down(&self) -> bool {
        match self {
            GridState::Empty => false,
            GridState::Station => true,
            GridState::LR => false,
            GridState::UD => true,
            GridState::LD => true,
            GridState::LU => false,
            GridState::RU => false,
            GridState::RD => true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
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

fn in_range(x: i32, y: i32) -> bool {
    x >= 0 && x < N as i32 && y >= 0 && y < N as i32
}

fn calc_income(
    people: &Vec<Person>,
    grid_dsu: &mut ac_library::Dsu,
    grid_state: &Vec<Vec<GridState>>,
) -> usize {
    let mut res = 0;
    for &p in people.iter() {
        'outer: for (dx1, dy1) in MANHATTAN_2_LIST {
            if in_range(p.home.x as i32 + dx1, p.home.y as i32 + dy1) {
                let p1 = Point::new(
                    (p.home.x as i32 + dx1) as usize,
                    (p.home.y as i32 + dy1) as usize,
                );
                if grid_state[p1.x][p1.y] == GridState::Station {
                    for (dx2, dy2) in MANHATTAN_2_LIST {
                        if in_range(p.work.x as i32 + dx2, p.work.y as i32 + dy2) {
                            let p2 = Point::new(
                                (p.work.x as i32 + dx2) as usize,
                                (p.work.y as i32 + dy2) as usize,
                            );
                            if grid_dsu.same(p1.to_idx(), p2.to_idx())
                                && grid_state[p2.x][p2.y] == GridState::Station
                            {
                                res += p.dist();
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }
    res
}

#[fastout]
fn output(ans: &Vec<((GridState, Point), (usize, usize))>) {
    assert!(ans.len() >= T);
    for i in 0..T {
        let ((s, pos), (money, income)) = ans[i];
        println!("# Turn: {}", i + 1);
        println!("# Money: {} Income: {}", money, income);
        if s == GridState::Empty {
            println!("-1");
        } else {
            println!("{} {} {}", s.output(), pos.x, pos.y);
        }
    }
}

fn main() {
    let time = time::Instant::now();

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

    fn solve(
        cnttry: usize,
        people: &Vec<Person>,
        m: usize,
        k_: usize,
        stations: &Vec<Point>,
        solver: SolverType,
    ) -> Vec<((GridState, Point), (usize, usize))> {
        let mut k = k_;

        let mut rng: StdRng = SeedableRng::seed_from_u64(cnttry as u64);
        let mut grid_to_peopleidx = vec![vec![collections::HashSet::new(); N]; N];
        {
            for i in 0..m {
                for (dx, dy) in MANHATTAN_2_LIST {
                    let nxh = people[i].home.x as i32 + dx;
                    let nyh = people[i].home.y as i32 + dy;
                    let nxw = people[i].work.x as i32 + dx;
                    let nyw = people[i].work.y as i32 + dy;
                    if in_range(nxh, nyh) {
                        grid_to_peopleidx[nxh as usize][nyh as usize].insert(i);
                    }
                    if in_range(nxw, nyw) {
                        grid_to_peopleidx[nxw as usize][nyw as usize].insert(i);
                    }
                }
            }
        }

        let mut build_todo = collections::VecDeque::new();
        let mut target_grid = vec![vec!['.'; N]; N];
        let is_station_pos = {
            let mut res = vec![vec![false; N]; N];
            for &p in stations.iter() {
                res[p.x][p.y] = true;
            }
            res
        };

        // 最初の 2 点を決める
        // 最初の所持金で建設可能なもののうち、収益が最も高いところに建てる
        {
            let mut best = (0, Point::new(!0, !0), Point::new(!0, !0));
            for i in 0..N {
                for j in 0..N {
                    let p = Point::new(i, j);
                    let mut cand = collections::BTreeMap::new();
                    for &i in &grid_to_peopleidx[i][j] {
                        let pp = &people[i];
                        if manhattan_distance(&pp.home, &p) <= 2 {
                            for (dx, dy) in MANHATTAN_2_LIST {
                                let nx = pp.work.x as i32 + dx;
                                let ny = pp.work.y as i32 + dy;
                                if in_range(nx, ny) {
                                    let q = Point::new(nx as usize, ny as usize);
                                    cand.entry(q).or_insert(0);
                                    *cand.get_mut(&q).unwrap() += pp.dist();
                                }
                            }
                        }
                        if manhattan_distance(&pp.work, &p) <= 2 {
                            for (dx, dy) in MANHATTAN_2_LIST {
                                let nx = pp.home.x as i32 + dx;
                                let ny = pp.home.y as i32 + dy;
                                if in_range(nx, ny) {
                                    let q = Point::new(nx as usize, ny as usize);
                                    cand.entry(q).or_insert(0);
                                    *cand.get_mut(&q).unwrap() += pp.dist();
                                }
                            }
                        }
                    }
                    for (q, income) in cand {
                        let score = income as i64
                            * (T as i64 - 1 - manhattan_distance(&p, &q) as i64)
                            - (2 * COST_STATION
                                + (manhattan_distance(&p, &q) - 1) as usize * COST_RAIL)
                                as i64;
                        if score > best.0
                            && 2 * COST_STATION
                                + (manhattan_distance(&p, &q) - 1) as usize * COST_RAIL
                                <= k
                        {
                            best = (score, p, q);
                        }
                    }
                }
            }
            if best.0 > 0 {
                build_todo.push_back((GridState::Station, best.1.x, best.1.y));
                build_todo.push_back((GridState::Station, best.2.x, best.2.y));

                // 01BFS: 今後駅が建つ予定ならそこを通りたい
                let mut grid_dist = vec![vec![INF; N]; N];
                let mut prv = vec![vec![Point::new(!0, !0); N]; N];
                let mut que = collections::VecDeque::new();
                que.push_back(best.1);
                grid_dist[best.1.x][best.1.y] = 0;
                prv[best.1.x][best.1.y] = best.1;

                while !que.is_empty() {
                    let q = que.pop_front().unwrap();
                    if q == best.2 {
                        break;
                    }
                    let mut cand = [q.left(), q.right(), q.up(), q.down()];
                    cand.shuffle(&mut rng);
                    for &r in &cand {
                        if !r.in_range() {
                            continue;
                        }
                        if is_station_pos[r.x][r.y] && grid_dist[r.x][r.y] > grid_dist[q.x][q.y] {
                            grid_dist[r.x][r.y] = grid_dist[q.x][q.y];
                            prv[r.x][r.y] = q;
                            que.push_front(r);
                        } else if grid_dist[r.x][r.y] > grid_dist[q.x][q.y] + 1 {
                            grid_dist[r.x][r.y] = grid_dist[q.x][q.y] + 1;
                            prv[r.x][r.y] = q;
                            que.push_back(r);
                        }
                    }
                }

                assert_ne!(grid_dist[best.2.x][best.2.y], INF);

                // 復元
                let mut now_pos = best.2;
                let mut prv_pos = best.2;
                while now_pos != best.1 {
                    let next_pos = prv[now_pos.x][now_pos.y];
                    assert_ne!(next_pos, Point::new(!0, !0));
                    if prv_pos != now_pos {
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
                        target_grid[now_pos.x][now_pos.y] = GridState::from_mask(mask).to_char();
                        build_todo.push_back((GridState::from_mask(mask), now_pos.x, now_pos.y));
                    }

                    prv_pos = now_pos;
                    now_pos = next_pos;
                }
                target_grid[best.1.x][best.1.y] = '#';
                target_grid[best.2.x][best.2.y] = '#';
            }
        }

        // どんどん駅をつなげていく
        let mut profit_table = vec![vec![0; N]; N];
        let mut cost = vec![vec![collections::BinaryHeap::new(); N]; N];
        let mut cand_pos = Vec::new();
        let mut connected_home = vec![false; m];
        let mut connected_work = vec![false; m];

        // 人の数と費用も考慮する
        fn calc_profit(
            p: &Point,
            profit_table: &Vec<Vec<usize>>,
            grid_to_peopleidx: &Vec<Vec<collections::HashSet<usize>>>,
            cost: &Vec<Vec<collections::BinaryHeap<Reverse<(u32, (Point, Direction))>>>>,
        ) -> i64 {
            profit_table[p.x][p.y] as i64 * 10000i64
                + grid_to_peopleidx[p.x][p.y].len() as i64 * 100i64
                - (cost[p.x][p.y]
                    .peek()
                    .or(Some(&Reverse((
                        INF as u32,
                        (Point::new(!0, !0), Direction::None),
                    ))))
                    .unwrap()
                    .0
                     .0 as i64)
        }

        {
            // 前の 2 要素を取得
            // pop しないより良い方法がありそうだけどまあ許容
            let sta1 = build_todo.pop_front().unwrap();
            let sta2 = build_todo.pop_front().unwrap();
            build_todo.push_front(sta2);
            build_todo.push_front(sta1);

            assert_eq!(sta1.0, GridState::Station);
            assert_eq!(sta2.0, GridState::Station);

            let sta1pos = Point::new(sta1.1, sta1.2);
            let sta2pos = Point::new(sta2.1, sta2.2);
            let mut candidx = collections::HashSet::<usize>::new();
            candidx.extend(&grid_to_peopleidx[sta1pos.x][sta1pos.y].clone());
            candidx.extend(&grid_to_peopleidx[sta2pos.x][sta2pos.y].clone());
            for &i in candidx.iter() {
                let pp = &people[i];
                if manhattan_distance(&pp.home, &sta1pos) <= 2
                    || manhattan_distance(&pp.home, &sta2pos) <= 2
                {
                    connected_home[i] = true;
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nx = pp.home.x as i32 + dx;
                        let ny = pp.home.y as i32 + dy;
                        if in_range(nx, ny) {
                            grid_to_peopleidx[nx as usize][ny as usize].remove(&i);
                        }
                    }
                }
                if manhattan_distance(&pp.work, &sta1pos) <= 2
                    || manhattan_distance(&pp.work, &sta2pos) <= 2
                {
                    connected_work[i] = true;
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nx = pp.work.x as i32 + dx;
                        let ny = pp.work.y as i32 + dy;
                        if in_range(nx, ny) {
                            grid_to_peopleidx[nx as usize][ny as usize].remove(&i);
                        }
                    }
                }
            }

            // BFS
            for &start in &[sta1pos, sta2pos] {
                let mut que = collections::VecDeque::new();
                let mut dist = vec![vec![INF; N]; N];
                que.push_back(start);
                dist[start.x][start.y] = 0;
                while !que.is_empty() {
                    let q = que.pop_front().unwrap();
                    for &nxt in &[q.left(), q.right(), q.up(), q.down()] {
                        if nxt.in_range()
                            && dist[nxt.x][nxt.y] == INF
                            && target_grid[nxt.x][nxt.y] == '.'
                        {
                            dist[nxt.x][nxt.y] = dist[q.x][q.y] + 1;
                            que.push_back(nxt);
                        }
                    }
                }
                for i in 0..N {
                    for j in 0..N {
                        if dist[i][j] != INF && target_grid[i][j] == '.' {
                            // いったん全方位いれちゃう
                            for &dir in &[
                                Direction::Left,
                                Direction::Right,
                                Direction::Up,
                                Direction::Down,
                            ] {
                                cost[i][j].push(Reverse((dist[i][j] as u32 - 1, (start, dir))));
                            }
                        } else if target_grid[i][j] != '.' && target_grid[i][j] != '#' {
                            cost[i][j].push(Reverse((0, (Point::new(!0, !0), Direction::None))));
                        }
                    }
                }
            }

            for i in 0..N {
                for j in 0..N {
                    let p = Point::new(i, j);
                    cand_pos.push(p);
                    for &id in &grid_to_peopleidx[i][j] {
                        let pp = &people[id];
                        assert!(!(connected_home[id] && connected_work[id]));
                        if manhattan_distance(&pp.home, &p) <= 2 && connected_work[id] {
                            profit_table[i][j] += pp.dist();
                        }
                        if manhattan_distance(&pp.work, &p) <= 2 && connected_home[id] {
                            profit_table[i][j] += pp.dist();
                        }
                    }

                    while !cost[i][j].is_empty() {
                        let (_, (q, d)) = cost[i][j].peek().unwrap().0;
                        if d == Direction::None {
                            break;
                        }
                        let p = match d {
                            Direction::Left => q.left(),
                            Direction::Right => q.right(),
                            Direction::Up => q.up(),
                            Direction::Down => q.down(),
                            _ => unreachable!(),
                        };
                        if !p.in_range() || target_grid[p.x][p.y] != '.' {
                            cost[i][j].pop();
                            continue;
                        }
                        break;
                    }
                }
            }
        }

        // 答えを出すパート
        let mut turn = 0;
        let mut income = 0;
        let mut grid_dsu = ac_library::Dsu::new(N * N);
        let mut grid_state = vec![vec![GridState::Empty; N]; N];

        // (output, ターン終了時の (money, income))
        let mut ans = vec![((GridState::Empty, Point::new(!0, !0)), (0, 0)); T];

        while turn < T {
            if !build_todo.is_empty() {
                turn += 1;

                let &(r, i, j) = build_todo.front().unwrap();
                let p = Point::new(i, j);
                assert!(p.in_range());
                assert_ne!(r, GridState::Empty);

                if r == GridState::Station && k >= COST_STATION {
                    k -= COST_STATION;
                } else if r != GridState::Station && k >= COST_RAIL {
                    k -= COST_RAIL;
                } else {
                    k += income;
                    ans[turn - 1] = ((GridState::Empty, Point::new(!0, !0)), (k, income));
                    continue;
                }

                build_todo.pop_front();
                grid_state[i][j] = r;

                if grid_state[i][j].can_conn_left()
                    && p.left().in_range()
                    && grid_state[p.left().x][p.left().y].can_conn_right()
                {
                    grid_dsu.merge(p.to_idx(), p.left().to_idx());
                }
                if grid_state[i][j].can_conn_right()
                    && p.right().in_range()
                    && grid_state[p.right().x][p.right().y].can_conn_left()
                {
                    grid_dsu.merge(p.to_idx(), p.right().to_idx());
                }
                if grid_state[i][j].can_conn_up()
                    && p.up().in_range()
                    && grid_state[p.up().x][p.up().y].can_conn_down()
                {
                    grid_dsu.merge(p.to_idx(), p.up().to_idx());
                }
                if grid_state[i][j].can_conn_down()
                    && p.down().in_range()
                    && grid_state[p.down().x][p.down().y].can_conn_up()
                {
                    grid_dsu.merge(p.to_idx(), p.down().to_idx());
                }

                income = calc_income(&people, &mut grid_dsu, &grid_state);

                k += income;

                ans[turn - 1] = ((r, p), (k, income));

                continue;
            }

            cand_pos.sort_unstable_by(|a, b| {
                let sca = calc_profit(a, &profit_table, &grid_to_peopleidx, &cost);
                let scb = calc_profit(b, &profit_table, &grid_to_peopleidx, &cost);

                if solver == SolverType::Type1 {
                    // これまでと同じように計算
                    return scb.cmp(&sca);
                }

                // 今後の収益性を判断
                let numrail_a = cost[a.x][a.y]
                    .peek()
                    .or(Some(&Reverse((
                        INF as u32,
                        (Point::new(!0, !0), Direction::None),
                    ))))
                    .unwrap()
                    .0
                     .0 as i64;
                let numrail_b = cost[b.x][b.y]
                    .peek()
                    .or(Some(&Reverse((
                        INF as u32,
                        (Point::new(!0, !0), Direction::None),
                    ))))
                    .unwrap()
                    .0
                     .0 as i64;

                let cost_a = numrail_a as i64 * COST_RAIL as i64 + COST_STATION as i64;
                let cost_b = numrail_b as i64 * COST_RAIL as i64 + COST_STATION as i64;

                let waitturn_a = (cost_a - k as i64 + income as i64 - 1).max(0) / income as i64;
                let waitturn_b = (cost_b - k as i64 + income as i64 - 1).max(0) / income as i64;

                let buildturn_a = 1 + numrail_a;
                let buildturn_b = 1 + numrail_b;

                let profit_a = profit_table[a.x][a.y] as i64;
                let profit_b = profit_table[b.x][b.y] as i64;

                let score_a = ((T - turn) as i64 - waitturn_a - buildturn_a) * profit_a - cost_a;
                let score_b = ((T - turn) as i64 - waitturn_b - buildturn_b) * profit_b - cost_b;

                if score_a > 0 || score_b > 0 {
                    // スコア大きいほうを返したいので逆順
                    score_b.cmp(&score_a)
                } else {
                    // これまでと同じ方法で計算
                    scb.cmp(&sca)
                }
            });

            let &p = cand_pos.first().unwrap();
            cand_pos.remove(0);

            // もしすでに駅が建ってれば何もしない
            if grid_state[p.x][p.y] == GridState::Station {
                continue;
            }
            // もしすでに線路がひかれていればそこに置くだけ
            else if target_grid[p.x][p.y] != '.' {
                target_grid[p.x][p.y] = '#';
                build_todo.push_back((GridState::Station, p.x, p.y));
            } else {
                // 01BFS: 今後駅が建つ予定ならそこを通りたい
                let mut grid_dist = vec![vec![(INF, INF); N]; N]; // 0: 駅を通る 01 スコア, 1: リアル距離
                let mut que = collections::VecDeque::new();
                let mut prv = vec![vec![Point::new(!0, !0); N]; N];
                que.push_back(p);
                grid_dist[p.x][p.y] = (0, 0);
                prv[p.x][p.y] = p;
                let mut target = Point::new(!0, !0);
                while !que.is_empty() {
                    let q = que.pop_front().unwrap();
                    if target_grid[q.x][q.y] == '#' && target == Point::new(!0, !0) {
                        target = q;
                    }
                    let mut cand = [q.left(), q.right(), q.up(), q.down()];
                    cand.shuffle(&mut rng);
                    for &r in &cand {
                        if !r.in_range()
                            || (target_grid[r.x][r.y] != '.' && target_grid[r.x][r.y] != '#')
                        {
                            continue;
                        }
                        if is_station_pos[r.x][r.y] && grid_dist[r.x][r.y].0 > grid_dist[q.x][q.y].0
                        {
                            grid_dist[r.x][r.y].0 = grid_dist[q.x][q.y].0;
                            grid_dist[r.x][r.y].1 = grid_dist[q.x][q.y].1 + 1;
                            prv[r.x][r.y] = q;
                            que.push_front(r);
                        } else if grid_dist[r.x][r.y].0 > grid_dist[q.x][q.y].0 + 1 {
                            grid_dist[r.x][r.y].0 = grid_dist[q.x][q.y].0 + 1;
                            grid_dist[r.x][r.y].1 = grid_dist[q.x][q.y].1 + 1;
                            prv[r.x][r.y] = q;
                            que.push_back(r);
                        }
                    }
                }
                // 到達不可能
                if target == Point::new(!0, !0) {
                    continue;
                }

                // 復元
                let mut now_pos = target;
                let mut prv_pos = target;
                while now_pos != p {
                    let next_pos = prv[now_pos.x][now_pos.y];
                    assert_ne!(next_pos, Point::new(!0, !0));

                    if target_grid[now_pos.x][now_pos.y] == '#' {
                        target_grid[now_pos.x][now_pos.y] = '#';
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
                        target_grid[now_pos.x][now_pos.y] = GridState::from_mask(mask).to_char();
                        build_todo.push_back((GridState::from_mask(mask), now_pos.x, now_pos.y));
                    }

                    prv_pos = now_pos;
                    now_pos = next_pos;
                }
                target_grid[p.x][p.y] = '#';
                target_grid[target.x][target.y] = '#';
                build_todo.push_back((GridState::Station, p.x, p.y));

                for i in 0..N {
                    for j in 0..N {
                        if target_grid[i][j] != '.' {
                            // どうせ Point は使われない
                            cost[i][j].push(Reverse((0, (Point::new(!0, !0), Direction::None))));
                        }
                        if grid_dist[i][j].1 != INF {
                            // もう全部つなげて下で pop させる
                            cost[i][j].push(Reverse((
                                grid_dist[i][j].1 as u32 - 1,
                                (p, Direction::Left),
                            )));
                            cost[i][j].push(Reverse((
                                grid_dist[i][j].1 as u32 - 1,
                                (p, Direction::Right),
                            )));
                            cost[i][j]
                                .push(Reverse((grid_dist[i][j].1 as u32 - 1, (p, Direction::Up))));
                            cost[i][j].push(Reverse((
                                grid_dist[i][j].1 as u32 - 1,
                                (p, Direction::Down),
                            )));
                        }
                        while !cost[i][j].is_empty() {
                            let (_, (q, d)) = cost[i][j].peek().unwrap().0;
                            if d == Direction::None {
                                break;
                            }
                            let p = match d {
                                Direction::Left => q.left(),
                                Direction::Right => q.right(),
                                Direction::Up => q.up(),
                                Direction::Down => q.down(),
                                _ => unreachable!(),
                            };
                            if !p.in_range() || target_grid[p.x][p.y] != '.' {
                                cost[i][j].pop();
                                continue;
                            }
                            break;
                        }
                    }
                }
            }

            // 新しく p に駅ができるので更新
            let candidx = grid_to_peopleidx[p.x][p.y].clone();
            for &i in candidx.iter() {
                let pp = &people[i];
                if manhattan_distance(&pp.home, &p) <= 2 {
                    connected_home[i] = true;
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nxh = pp.home.x as i32 + dx;
                        let nyh = pp.home.y as i32 + dy;
                        if in_range(nxh, nyh) {
                            grid_to_peopleidx[nxh as usize][nyh as usize].remove(&i);
                        }
                    }
                }
                if manhattan_distance(&pp.work, &p) <= 2 {
                    connected_work[i] = true;
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nxw = pp.work.x as i32 + dx;
                        let nyw = pp.work.y as i32 + dy;
                        if in_range(nxw, nyw) {
                            grid_to_peopleidx[nxw as usize][nyw as usize].remove(&i);
                        }
                    }
                }
            }
            for &i in candidx.iter() {
                let pp = &people[i];
                if manhattan_distance(&pp.home, &p) <= 2 {
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nxh = pp.home.x as i32 + dx;
                        let nyh = pp.home.y as i32 + dy;
                        let nxw = pp.work.x as i32 + dx;
                        let nyw = pp.work.y as i32 + dy;
                        if in_range(nxh, nyh) && connected_work[i] {
                            profit_table[nxh as usize][nyh as usize] -= pp.dist();
                        }
                        if in_range(nxw, nyw)
                            && grid_to_peopleidx[nxw as usize][nyw as usize].contains(&i)
                        {
                            profit_table[nxw as usize][nyw as usize] += pp.dist();
                        }
                    }
                }
                if manhattan_distance(&pp.work, &p) <= 2 {
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nxh = pp.home.x as i32 + dx;
                        let nyh = pp.home.y as i32 + dy;
                        let nxw = pp.work.x as i32 + dx;
                        let nyw = pp.work.y as i32 + dy;
                        if in_range(nxw, nyw) && connected_home[i] {
                            profit_table[nxw as usize][nyw as usize] -= pp.dist();
                        }
                        if in_range(nxh, nyh)
                            && grid_to_peopleidx[nxh as usize][nyh as usize].contains(&i)
                        {
                            profit_table[nxh as usize][nyh as usize] += pp.dist();
                        }
                    }
                }
            }
        }

        ans
    }

    let mut best = (0, Vec::new());
    let mut prv_sta = Vec::new();
    let mut cnttry = 0;

    // 余裕をもって 2300ms で終了
    while time.elapsed().as_millis() < 2300 {
        let solver = if time.elapsed().as_millis() < 1500 {
            SolverType::Type1
        } else {
            SolverType::Type2
        };

        let ans = solve(cnttry, &people, m, k, &prv_sta, solver);
        eprintln!("#{}: {}ms", cnttry, time.elapsed().as_millis());

        // 駅一覧を取得
        prv_sta = {
            let mut res = Vec::new();
            for &((s, p), _) in &ans {
                if s == GridState::Station {
                    res.push(p);
                }
            }
            res
        };

        let score = {
            let mut res = 0;
            for i in 0..T {
                res = res.max(ans[i].1 .0 + ans[i].1 .1 * (T - i - 1));
            }
            res
        };

        if score > best.0 {
            best = (score, ans);
        }
        cnttry += 1;
    }

    // 最終的に待ったほうがいいなら Revert
    for t in (0..T).rev() {
        if best.1[T - 1].1 .0 < best.1[t].1 .0 + best.1[t].1 .1 * (T - t - 1) {
            for x in t + 1..T {
                best.1[x] = (
                    (GridState::Empty, Point::new(!0, !0)),
                    (best.1[t].1 .0 + best.1[t].1 .1 * (x - t), best.1[t].1 .1),
                );
            }
        }
    }
    eprintln!("Revert: {}ms", time.elapsed().as_millis());

    output(&best.1);
    eprintln!("Output: {}ms", time.elapsed().as_millis());
}
