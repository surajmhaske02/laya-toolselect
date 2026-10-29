#!/usr/bin/env bash
# Download MetaTool and BFCL v4 from GitHub and build benchmarks/data/{metatool,bfcl,notool}.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
raw="$here/raw"; mkdir -p "$raw/metatool" "$raw/bfcl"
M=https://raw.githubusercontent.com/HowieHwong/MetaTool/master/dataset
curl -fsSL -o "$raw/metatool/plugin_des.json" "$M/plugin_des.json"
curl -fsSL -o "$raw/metatool/all_clean_data.csv" "$M/data/all_clean_data.csv"
B=https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data
for f in simple_python multiple live_simple live_multiple irrelevance; do
  curl -fsSL -o "$raw/bfcl/BFCL_v4_$f.json" "$B/BFCL_v4_$f.json"
done
for f in simple_python multiple live_simple live_multiple; do
  curl -fsSL -o "$raw/bfcl/ans_$f.json" "$B/possible_answer/BFCL_v4_$f.json"
done
python "$here/prepare.py" --metatool "$raw/metatool" --bfcl "$raw/bfcl" --out "$here/data"
