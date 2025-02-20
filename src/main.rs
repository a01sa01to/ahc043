extern crate rand;
use proconio::{fastout, input};
use rand::seq::SliceRandom;
use std::{collections, fmt};

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

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
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
    for i in 0..ans.len() {
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
        people: &Vec<Person>,
        m: usize,
        k_: usize,
    ) -> Vec<((GridState, Point), (usize, usize))> {
        let mut k = k_;

        let mut rng = rand::thread_rng();
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
                        if income > best.0
                            && 2 * COST_STATION + manhattan_distance(&p, &q) as usize * COST_RAIL
                                <= k
                        {
                            best = (income, p, q);
                        }
                    }
                }
            }
            if best.0 > 0 {
                build_todo.push_back((GridState::Station, best.1.x, best.1.y));
                build_todo.push_back((GridState::Station, best.2.x, best.2.y));
                // TODO: もっと良い方法があるはず (今後駅が建つところを通ったほうがよさそう) だが適当にやる
                let mut now_pos = best.1;
                let mut prv_pos = best.1;
                while now_pos != best.2 {
                    let mut next_pos = now_pos;
                    let mut cand = vec![
                        now_pos.left(),
                        now_pos.right(),
                        now_pos.up(),
                        now_pos.down(),
                    ];
                    cand.shuffle(&mut rng);
                    for &q in &cand {
                        if q.in_range()
                            && manhattan_distance(&q, &best.2)
                                < manhattan_distance(&now_pos, &best.2)
                        {
                            next_pos = q;
                            break;
                        }
                    }
                    assert_ne!(next_pos, now_pos);

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
        let mut cost = vec![vec![0; N]; N];
        let mut cand_pos = Vec::new();
        let mut connected_home = vec![false; m];
        let mut connected_work = vec![false; m];

        // 人の数と費用も考慮する
        fn calc_profit(
            p: &Point,
            profit_table: &Vec<Vec<usize>>,
            grid_to_peopleidx: &Vec<Vec<collections::HashSet<usize>>>,
            cost: &Vec<Vec<u32>>,
        ) -> i64 {
            profit_table[p.x][p.y] as i64 * 10000i64
                + grid_to_peopleidx[p.x][p.y].len() as i64 * 100i64
                - cost[p.x][p.y] as i64
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

            for i in 0..N {
                for j in 0..N {
                    let p = Point::new(i, j);
                    cand_pos.push(p);
                    cost[i][j] =
                        manhattan_distance(&sta1pos, &p).min(manhattan_distance(&sta2pos, &p));
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
                }
            }

            // 降順にしたいので profit_table[b].cmp[a] にする
            cand_pos.sort_unstable_by(|a, b| {
                calc_profit(b, &profit_table, &grid_to_peopleidx, &cost).cmp(&calc_profit(
                    a,
                    &profit_table,
                    &grid_to_peopleidx,
                    &cost,
                ))
            });
            while !cand_pos.is_empty()
                && calc_profit(
                    cand_pos.last().unwrap(),
                    &profit_table,
                    &grid_to_peopleidx,
                    &cost,
                ) <= 0
            {
                cand_pos.pop();
            }
        }

        while !cand_pos.is_empty() {
            let &p = cand_pos.first().unwrap();
            cand_pos.remove(0);

            // もしすでに線路がひかれていればそこに置くだけ
            if target_grid[p.x][p.y] != '.' {
                target_grid[p.x][p.y] = '#';
                build_todo.push_back((GridState::Station, p.x, p.y));
            } else {
                // BFS
                // 01 にする必要はない: 駅にぶつかったら終了するので既存の線路は使わない？
                let mut grid_dist = vec![vec![INF; N]; N];
                let mut que = collections::VecDeque::new();
                que.push_back(p);
                grid_dist[p.x][p.y] = 0;
                let mut target = Point::new(!0, !0);
                while !que.is_empty() {
                    let q = que.pop_front().unwrap();
                    if target_grid[q.x][q.y] == '#' && target == Point::new(!0, !0) {
                        target = q;
                    }
                    for &r in &[q.left(), q.right(), q.up(), q.down()] {
                        if !r.in_range()
                            || grid_dist[r.x][r.y] != INF
                            || (target_grid[r.x][r.y] != '.' && target_grid[r.x][r.y] != '#')
                        {
                            continue;
                        }
                        grid_dist[r.x][r.y] = grid_dist[q.x][q.y] + 1;
                        que.push_back(r);
                    }
                }
                // 到達不可能
                if target == Point::new(!0, !0) {
                    // cand_pos.push(p);
                    continue;
                }

                // 復元
                let mut now_pos = target;
                let mut prv_pos = target;
                while now_pos != p {
                    let mut next_pos = now_pos;
                    let mut cand = vec![
                        now_pos.left(),
                        now_pos.right(),
                        now_pos.up(),
                        now_pos.down(),
                    ];
                    cand.shuffle(&mut rng);
                    for &q in &cand {
                        if q.in_range()
                            && grid_dist[q.x][q.y] + 1 == grid_dist[now_pos.x][now_pos.y]
                        {
                            next_pos = q;
                            break;
                        }
                    }
                    assert_ne!(next_pos, now_pos);

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
                        if grid_dist[i][j] != INF {
                            cost[i][j] = cost[i][j].min(grid_dist[i][j] as u32);
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
                        let nxw = pp.work.x as i32 + dx;
                        let nyw = pp.work.y as i32 + dy;
                        if in_range(nxh, nyh) {
                            grid_to_peopleidx[nxh as usize][nyh as usize].remove(&i);
                            if connected_work[i] {
                                profit_table[nxh as usize][nyh as usize] -= pp.dist();
                            }
                        }
                        if in_range(nxw, nyw)
                            && grid_to_peopleidx[nxw as usize][nyw as usize].contains(&i)
                        {
                            profit_table[nxw as usize][nyw as usize] += pp.dist();
                        }
                    }
                }
                if manhattan_distance(&pp.work, &p) <= 2 {
                    connected_work[i] = true;
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nxh = pp.home.x as i32 + dx;
                        let nyh = pp.home.y as i32 + dy;
                        let nxw = pp.work.x as i32 + dx;
                        let nyw = pp.work.y as i32 + dy;
                        if in_range(nxw, nyw) {
                            grid_to_peopleidx[nxw as usize][nyw as usize].remove(&i);
                            if connected_home[i] {
                                profit_table[nxw as usize][nyw as usize] -= pp.dist();
                            }
                        }
                        if in_range(nxh, nyh)
                            && grid_to_peopleidx[nxh as usize][nyh as usize].contains(&i)
                        {
                            profit_table[nxh as usize][nyh as usize] += pp.dist();
                        }
                    }
                }
            }

            cand_pos.sort_unstable_by(|a, b| {
                (calc_profit(b, &profit_table, &grid_to_peopleidx, &cost)).cmp(&calc_profit(
                    a,
                    &profit_table,
                    &grid_to_peopleidx,
                    &cost,
                ))
            });
            while !cand_pos.is_empty()
                && calc_profit(
                    cand_pos.last().unwrap(),
                    &profit_table,
                    &grid_to_peopleidx,
                    &cost,
                ) <= 0
            {
                cand_pos.pop();
            }
        }

        let stations = {
            let mut res = Vec::new();
            for i in 0..N {
                for j in 0..N {
                    if target_grid[i][j] == '#' {
                        res.push(Point::new(i, j));
                    }
                }
            }
            res
        };

        let people2sta = {
            let mut res = vec![(Vec::new(), Vec::new()); m];
            for (i, p) in people.iter().enumerate() {
                for (j, s) in stations.iter().enumerate() {
                    for (dx, dy) in MANHATTAN_2_LIST {
                        let nx = s.x as i32 + dx;
                        let ny = s.y as i32 + dy;
                        let po = Point::new(nx as usize, ny as usize);
                        if po.in_range() && p.home == po {
                            res[i].0.push(j);
                        }
                        if po.in_range() && p.work == po {
                            res[i].1.push(j);
                        }
                    }
                }
            }
            res
        };

        // 答えを出すパート
        let mut turn = 0;
        let mut income = 0;
        let mut nconnected_peopleidx = collections::HashSet::new();
        let mut grid_dsu = ac_library::Dsu::new(N * N);
        let mut grid_state = vec![vec![GridState::Empty; N]; N];

        // (output, ターン終了時の (money, income))
        let mut ans = vec![((GridState::Empty, Point::new(!0, !0)), (0, 0)); T];

        for i in 0..m {
            if !people2sta[i].0.is_empty() && !people2sta[i].1.is_empty() {
                nconnected_peopleidx.insert(i);
            }
        }

        while turn < T {
            turn += 1;

            if build_todo.is_empty() {
                for x in turn..=T {
                    k += income;
                    ans[x - 1] = ((GridState::Empty, Point::new(!0, !0)), (k, income));
                }
                break;
            }

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
        }
        ans
    }

    let mut ans = solve(&people, m, k);

    // 最終的に待ったほうがいいなら Revert
    for t in (0..T).rev() {
        if ans[T - 1].1 .0 < ans[t].1 .0 + ans[t].1 .1 * (T - t) {
            for x in t + 1..T {
                ans[x] = (
                    (GridState::Empty, Point::new(!0, !0)),
                    (ans[t].1 .0 + ans[t].1 .1 * (x - t), ans[t].1 .1),
                );
            }
        }
    }

    output(&ans);
}
