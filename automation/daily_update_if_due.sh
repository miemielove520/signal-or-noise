#!/bin/bash
# Catch-up guard for launchd. It runs the full pipeline at most once per weekday.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS="$ROOT/outputs/dashboard/last_run_status.json"
TODAY="$(date '+%Y-%m-%d')"
WEEKDAY="$(date '+%u')"
NOW_HHMM="$(date '+%H%M')"

# Monday-Friday only, and never before the intended 15:00 local schedule.
if [ "$WEEKDAY" -gt 5 ] || [ "$NOW_HHMM" -lt 1500 ]; then
  exit 0
fi

# Any completed attempt today counts as handled. Failed runs require manual review,
# rather than repeatedly consuming API quota every 30 minutes.
if [ -f "$STATUS" ] && grep -q "\"finished_at\":\"$TODAY" "$STATUS"; then
  exit 0
fi

exec /bin/bash "$ROOT/daily_update.sh" >> "$ROOT/outputs/daily_update.log" 2>&1
