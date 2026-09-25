#!/bin/bash
# 每天自动运行：分析关注列表里的每只股票，然后生成网页看板。
# Daily auto-run: analyze each ticker in watchlist.txt, then build the HTML dashboard.
#
# 完整功能已配置好（基本面趋势 + Tiingo 数据）。
# 密钥和个人信息放在 daily_update.env（不进 git）。首次使用：
#   cp daily_update.env.example daily_update.env  然后填入你的 token。
cd "$(dirname "$0")" || exit 1

mkdir -p outputs
RUN_MARKER="outputs/.daily_update_started"
touch "$RUN_MARKER"

if [ -f daily_update.env ]; then
  # shellcheck disable=SC1091
  . ./daily_update.env
else
  echo "warning: daily_update.env not found — API keys/SEC_USER_AGENT unset, data fetch may degrade" >&2
fi

# 日志轮转：超过 1MB 只保留最后 4000 行（launchd 以追加模式写同一文件，原地截断是安全的）。
LOG_FILE="outputs/daily_update.log"
if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE")" -gt 1048576 ]; then
  tail -n 4000 "$LOG_FILE" > "$LOG_FILE.tmp" && cat "$LOG_FILE.tmp" > "$LOG_FILE" && rm -f "$LOG_FILE.tmp"
fi

if [ -x .venv/bin/python ]; then
  PYTHON="$PWD/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# 失败追踪：任何一步失败都记下来，最后写进状态文件让看板亮红条 + 非零退出。
# 之前每步 `|| echo` 把错误吞掉、退出码恒 0，坏了没人知道（launchd 那次就是这么静默失败的）。
FAILURES=""
note_fail() {
  FAILURES="${FAILURES:+$FAILURES; }$1"
  echo "  (failed: $1)"
}

analyze_list() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r ticker || [ -n "$ticker" ]; do
    ticker="$(echo "$ticker" | tr -d '[:space:]')"
    [ -z "$ticker" ] && continue
    case "$ticker" in \#*) continue ;; esac
    echo "[$(date '+%Y-%m-%d %H:%M')] analyzing $ticker"
    "$PYTHON" run.py "$ticker" --no-peers || note_fail "analyze $ticker"
  done < "$file"
}

analyze_list market.txt      # 大盘参考 VOO / QQQM
analyze_list watchlist.txt   # 我的持仓

# 板块 ETF：下载价格供看板算「板块轮动 / 资金流」（sectors.txt 一行多个用空格分隔）。
if [ -f sectors.txt ]; then
  for t in $(grep -vE '^[[:space:]]*#' sectors.txt); do
    echo "[$(date '+%Y-%m-%d %H:%M')] sector $t"
    "$PYTHON" run.py "$t" --no-snapshot --no-peers >/dev/null 2>&1 || note_fail "sector $t"
  done
fi

# Scan the candidate universe and retain a daily comparison journal.
echo "[$(date '+%Y-%m-%d %H:%M')] scanning candidates_universe.txt for candidates"
CANDIDATE_TICKERS="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' candidates_universe.txt | tr '\n' ' ')"
if [ -n "$CANDIDATE_TICKERS" ]; then
  "$PYTHON" scan.py $CANDIDATE_TICKERS --journal || note_fail "candidate scan"
fi

# Rebalance the $1,000 paper portfolio and record its mark-to-market value.
echo "[$(date '+%Y-%m-%d %H:%M')] paper rebalance + mark-to-market"
"$PYTHON" paper.py --update-state --initial-cash 1000 --allow-near-watchlist || note_fail "paper rebalance"
"$PYTHON" paper_track.py || note_fail "paper track"

# Refresh due signal reviews.
"$PYTHON" review_due.py || note_fail "signal review due"

# Key outputs must be newer than this run's start marker.
check_fresh() {
  local file="$1" label="$2"
  if [ ! -f "$file" ] || [ ! "$file" -nt "$RUN_MARKER" ]; then
    note_fail "$label 未在本次运行中更新"
  fi
}
check_fresh outputs/scans/latest/high_probability_scan.csv "候选扫描"
check_fresh outputs/paper_latest/paper_performance.json "模拟盘"
check_fresh outputs/paper_history/paper_journal.md "模拟盘审计"
check_fresh outputs/signal_review/human_vs_model.md "人机记分板"
FIRST_TICKER="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' watchlist.txt 2>/dev/null | head -1 | tr -d '[:space:]')"
[ -n "$FIRST_TICKER" ] && check_fresh "outputs/real_ticker/$FIRST_TICKER/analysis_result.json" "持仓分析($FIRST_TICKER)"

# Write status once for page rendering and again after the dashboard completes.
mkdir -p outputs/dashboard
write_status() {
  if [ -n "$FAILURES" ]; then RUN_OK=false; else RUN_OK=true; fi
  SAFE_FAILURES="$(printf '%s' "$FAILURES" | tr '"\\' "''")"
  printf '{"finished_at":"%s","ok":%s,"failures":"%s"}\n' \
    "$(date '+%Y-%m-%d %H:%M')" "$RUN_OK" "$SAFE_FAILURES" > outputs/dashboard/last_run_status.json
}

write_status
if "$PYTHON" build_dashboard.py; then
  echo "[$(date '+%Y-%m-%d %H:%M')] dashboard updated: outputs/dashboard/index.html"
else
  note_fail "build dashboard"
fi
write_status

if [ -n "$FAILURES" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M')] RUN HAD FAILURES: $FAILURES" >&2
  exit 1
fi
