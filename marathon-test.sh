#!/bin/bash

export RUST_BACKTRACE=1
cargo build
if [ $? -ne 0 ]; then
  echo "Build failed..."
  exit 1
fi

cd tools
# ../a.out < in/0.txt 1> out/0.txt 2> err/0.txt
# cargo run --release --bin vis in/0.txt out/0.txt > res/0.txt
echo Running Test 0...
# cargo run --release --bin tester ../target/debug/a < in/0.txt 1> out/0.txt 2> err/0.txt
./target/release/tester ../target/debug/a < in/0.txt 1> out/0.txt 2> err/0.txt
# echo Running visualizer...
# cargo run --release --bin vis in/0.txt out/0.txt &> res/0.txt
cat err/0.txt
for i in {1..100}; do
  echo Running Test $i...
  # ../a.out < in/$i.txt 1> out/$i.txt 2> err/$i.txt
  # cargo run --release --bin vis in/$i.txt out/$i.txt > res/$i.txt
  # cargo run --release --bin tester ../target/debug/a < in/$i.txt 1> out/$i.txt 2> err/$i.txt
  ./target/release/tester ../target/debug/a < in/$i.txt 1> out/$i.txt 2> err/$i.txt
  cat err/$i.txt
  # echo Running visualizer...
  # cargo run --release --bin vis in/$i.txt out/$i.txt &> res/$i.txt
  # cat res/$i.txt
done
cd ..
python3 avg.py


# Todo
# 1. Generator Rustの{:04}を変える
# 2. まずはinputを0～100まで作る
# 3. out,res,errディレクトリをつくる
# 4. 上のコードを適切にする
# 5. avg.pyのpathを変える