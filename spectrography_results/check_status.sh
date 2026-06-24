#!/usr/bin/env bash
# Spectrography batch status checker
# Launched: 2026-06-20 ~19:56 UTC
# PIDs: Groq=40046, Mistral=40050, NIM=40064

GROQ_PID=40046
MISTRAL_PID=40050
NIM_PID=40064

LOG_DIR="$(cd "$(dirname "$0")" && pwd)"

check_process() {
    local name="$1" pid="$2" log="$3" total="$4"

    echo "--- $name (PID $pid, $total evals) ---"

    if ps -p "$pid" > /dev/null 2>&1; then
        local elapsed
        elapsed=$(ps -p "$pid" -o etime= 2>/dev/null | xargs)
        echo "  Status:    RUNNING (elapsed: $elapsed)"
    else
        echo "  Status:    FINISHED / EXITED"
    fi

    if [[ -f "$log" ]]; then
        local progress_count error_count warning_count checkpoint_count

        progress_count=$(grep -c 'progress=' "$log" || true)
        error_count=$(grep -ci 'error\|traceback\|exception' "$log" || true)
        warning_count=$(grep -c 'warning' "$log" || true)
        checkpoint_count=$(grep -c 'checkpoint' "$log" || true)

        echo "  Progress:  $progress_count / $total evaluations logged"
        echo "  Errors:    $error_count"
        echo "  Warnings:  $warning_count"
        echo "  Checkpts:  $checkpoint_count"

        echo "  Last 3 lines:"
        tail -3 "$log" | sed 's/^/    /'
    else
        echo "  Log file not found: $log"
    fi
    echo ""
}

echo "=== Spectrography Batch Status -- $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo ""

check_process "Groq Batch (8 models)"    "$GROQ_PID"    "$LOG_DIR/groq_batch.log"    648
check_process "Mistral Batch (13 models)" "$MISTRAL_PID" "$LOG_DIR/mistral_batch.log" 1053
check_process "NIM Batch (52 models)"     "$NIM_PID"     "$LOG_DIR/nim_batch.log"     4212

echo "--- Overall ---"
running=0
for pid in $GROQ_PID $MISTRAL_PID $NIM_PID; do
    ps -p "$pid" > /dev/null 2>&1 && running=$((running + 1))
done
echo "  Processes running: $running / 3"

total_progress=0
for log in "$LOG_DIR/groq_batch.log" "$LOG_DIR/mistral_batch.log" "$LOG_DIR/nim_batch.log"; do
    if [[ -f "$log" ]]; then
        count=$(grep -c 'progress=' "$log" || true)
        total_progress=$((total_progress + count))
    fi
done
echo "  Total evaluations: $total_progress / 5913"
echo ""
