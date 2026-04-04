#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  tune.sh  –  Auto-tunes matmul_tiled_2D_coarse_vec using Pre-Filtered configs
# ─────────────────────────────────────────────────────────────────────────────

SOURCE="finetune_kernel.cu"
EXE="/tmp/matmul_tune_$$"
N=2048
TIMEOUT_SEC=15

BEST_TIME_US=999999999
BEST_TIME_MS="9999.99"
BEST_PARAMS=""

is_numeric() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

echo "Generating valid configurations via Python..."
# Run the python script and store the output in an array/string
CONFIGS=$(python3 generate_configs.py)

# Count how many valid configs we actually have
CONFIG_COUNT=$(echo "$CONFIGS" | wc -l)

echo "Found $CONFIG_COUNT valid configurations to test."
echo "Starting Auto-Tuning  (N=$N, timeout=${TIMEOUT_SEC}s per config)"
printf "%-6s %-6s %-6s %-6s %-6s %-12s | %-9s | %s\n" \
       BM BN BK TM TN BLOCK_SIZE COMPUTE_T "Time(ms)"
echo "------------------------------------------------------------"

# Loop strictly over the pre-filtered configurations
while read -r BM BN BK TM TN BLOCK_SIZE COMPUTE_THREADS; do
    
    # Skip empty lines just in case
    [ -z "$BM" ] && continue

    COMPILE_ERR=$(mktemp /tmp/nvcc_err.XXXXXX)
    
    nvcc -O3 \
         -D_BM=$BM -D_BN=$BN -D_BK=$BK \
         -D_TM=$TM -D_TN=$TN \
         -D_BLOCK_SIZE=$BLOCK_SIZE \
         "$SOURCE" -o "$EXE" -lcublas 2>"$COMPILE_ERR"
    
    COMPILE_STATUS=$?
    rm -f "$COMPILE_ERR"

    if (( COMPILE_STATUS != 0 )); then 
        printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | COMPILE FAILED\n" \
               $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS
        continue
    fi

    OUT=$(mktemp /tmp/kern_out.XXXXXX)
    ERR=$(mktemp /tmp/kern_err.XXXXXX)

    timeout "$TIMEOUT_SEC" "$EXE" "$N" >"$OUT" 2>"$ERR"
    RUN_STATUS=$?

    RAW=$(grep -oE '^[0-9]+(\.[0-9]+)?$' "$OUT" | head -n1)
    rm -f "$OUT" "$ERR" "$EXE"

    if (( RUN_STATUS == 124 )); then 
        printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | TIMEOUT\n" \
               $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS
        continue 
    fi
    
    if [ -z "$RAW" ] || ! is_numeric "$RAW"; then 
        printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | CRASH/EMPTY\n" \
               $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS
        continue 
    fi

    if [ "$RAW" = "0" ] || [ "$RAW" = "0.0" ]; then 
        printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | VERIFY/EXEC FAILED\n" \
               $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS
        continue 
    fi

    # Record Success
    printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | %s ms\n" \
           $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS "$RAW"

    RESULT_US=$(awk -v r="$RAW" 'BEGIN { printf "%d", r * 1000 }')
    if (( RESULT_US < BEST_TIME_US && RESULT_US > 0 )); then
      BEST_TIME_US=$RESULT_US
      BEST_TIME_MS=$RAW
      BEST_PARAMS="BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN"
    fi

done <<< "$CONFIGS"

echo "------------------------------------------------------------"
if [ -n "$BEST_PARAMS" ]; then
    echo "WINNER : $BEST_PARAMS"
    echo "  Time : ${BEST_TIME_MS} ms"
else
    echo "No valid configurations found."
fi

rm -f "$EXE"