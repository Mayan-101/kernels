#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  tune.sh  –  Auto-tunes matmul_tiled_2D_coarse_vec
#
#  Parameter roles
#  ───────────────
#   BM, BN      : output tile size per CTA
#   BK          : phase depth along K
#   TM, TN      : per-thread sub-tile  (compute granularity)
#   BLOCK_SIZE  : "virtual loading threads" → controls stride_A / stride_B
#                 INDEPENDENT of the actual launch thread count (BM/TM)*(BN/TN)
#
#  Validity rules (checked in bash, zero compile cost)
#  ────────────────────────────────────────────────────
#  [Compute]
#    1. COMPUTE_THREADS = (BM/TM)*(BN/TN) ∈ [32, 1024]
#    2. BM % TM == 0,  BN % TN == 0
#  [Vectorised loads]
#    3. BK % 4 == 0,  BN % 4 == 0          (float4 alignment)
#    4. BLOCK_SIZE % (BK/4) == 0            (stride_A is an integer)
#    5. BLOCK_SIZE % (BN/4) == 0            (stride_B is an integer)
#    6. BM % stride_A == 0                  (A offset-loop tiles BM exactly)
#    7. BK % stride_B == 0                  (B offset-loop tiles BK exactly)
#  [Shared memory]
#    8. (BK*BM + BK*BN)*4 ≤ 49152 bytes    (fits in 48 KB smem)
# ─────────────────────────────────────────────────────────────────────────────

SOURCE="finetune_kernel.cu"
EXE="/tmp/matmul_tune_$$"
N=2048
TIMEOUT_SEC=15

BEST_TIME_US=999999999
BEST_TIME_MS="9999.99"
BEST_PARAMS=""

is_numeric() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

echo "Starting Auto-Tuning  (N=$N, timeout=${TIMEOUT_SEC}s per config)"
printf "%-6s %-6s %-6s %-6s %-6s %-12s | %-9s | %s\n" \
       BM BN BK TM TN BLOCK_SIZE COMPUTE_T "Time(ms)"
echo "------------------------------------------------------------"

# ── search space ─────────────────────────────────────────────────────────────
# BM / BN : output tile per CTA
for BM in 32 64 128; do
for BN in 32 64 128; do

  # BK : phase depth (must be multiple of 4 for float4)
  for BK in 8 16 32; do
    if (( BK % 4 != 0 )); then continue; fi

    # TM / TN : per-thread sub-tile
    for TM in 4 8; do
    for TN in 4 8; do

      # ── [Compute] validity ─────────────────────────────────────────────
      if (( BM % TM != 0 || BN % TN != 0 )); then continue; fi

      COMPUTE_THREADS=$(( (BM / TM) * (BN / TN) ))
      if (( COMPUTE_THREADS < 32 || COMPUTE_THREADS > 1024 )); then continue; fi

      # ── BLOCK_SIZE : independent loading-granularity parameter ────────
      # Try values that are powers-of-2 multiples of warp size; they must
      # also satisfy the stride divisibility rules checked inside the loop.
      for BLOCK_SIZE in 32 64 128 256 512; do

        # ── [Vectorised loads] validity ──────────────────────────────────

        # BN % 4 check (needed for float4 along BN)
        if (( BN % 4 != 0 )); then continue; fi

        # stride_A = BLOCK_SIZE / (BK/4)  →  BLOCK_SIZE must be divisible by (BK/4)
        BK_DIV4=$(( BK / 4 ))
        if (( BLOCK_SIZE % BK_DIV4 != 0 )); then continue; fi
        STRIDE_A=$(( BLOCK_SIZE / BK_DIV4 ))

        # stride_B = BLOCK_SIZE / (BN/4)  →  BLOCK_SIZE must be divisible by (BN/4)
        BN_DIV4=$(( BN / 4 ))
        if (( BLOCK_SIZE % BN_DIV4 != 0 )); then continue; fi
        STRIDE_B=$(( BLOCK_SIZE / BN_DIV4 ))

        # stride_A must tile BM exactly
        if (( STRIDE_A == 0 || BM % STRIDE_A != 0 )); then continue; fi

        # stride_B must tile BK exactly
        if (( STRIDE_B == 0 || BK % STRIDE_B != 0 )); then continue; fi

        # ── [Shared memory] check ────────────────────────────────────────
        SMEM=$(( (BK * BM + BK * BN) * 4 ))
        if (( SMEM > 49152 )); then continue; fi

        # ── Compile ──────────────────────────────────────────────────────
        COMPILE_ERR=$(mktemp /tmp/nvcc_err.XXXXXX)
        nvcc -O3 \
             -D_BM=$BM -D_BN=$BN -D_BK=$BK \
             -D_TM=$TM -D_TN=$TN \
             -D_BLOCK_SIZE=$BLOCK_SIZE \
             "$SOURCE" -o "$EXE" 2>"$COMPILE_ERR"
        COMPILE_STATUS=$?
        rm -f "$COMPILE_ERR"

        if (( COMPILE_STATUS != 0 )); then continue; fi

        # ── Run with timeout ─────────────────────────────────────────────
        OUT=$(mktemp /tmp/kern_out.XXXXXX)
        ERR=$(mktemp /tmp/kern_err.XXXXXX)

        timeout "$TIMEOUT_SEC" "$EXE" "$N" >"$OUT" 2>"$ERR"
        RUN_STATUS=$?

        RAW=$(grep -oE '^[0-9]+(\.[0-9]+)?$' "$OUT" | head -n1)
        rm -f "$OUT" "$ERR" "$EXE"

        # timeout / launch failure / kernel returned "0"
        if (( RUN_STATUS == 124 )); then continue; fi
        if [ -z "$RAW" ] || [ "$RAW" = "0" ] || ! is_numeric "$RAW"; then continue; fi

        # ── Record ───────────────────────────────────────────────────────
        printf "%-6d %-6d %-6d %-6d %-6d %-12d | %-9d | %s ms\n" \
               $BM $BN $BK $TM $TN $BLOCK_SIZE $COMPUTE_THREADS "$RAW"

        RESULT_US=$(awk -v r="$RAW" 'BEGIN { printf "%d", r * 1000 }')
        if (( RESULT_US < BEST_TIME_US && RESULT_US > 0 )); then
          BEST_TIME_US=$RESULT_US
          BEST_TIME_MS=$RAW
          BEST_PARAMS="BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN BLOCK_SIZE=$BLOCK_SIZE (compute_threads=$COMPUTE_THREADS)"
        fi

      done  # BLOCK_SIZE
    done; done  # TN TM
  done  # BK
done; done  # BN BM

echo "------------------------------------------------------------"
if [ -n "$BEST_PARAMS" ]; then
    echo "WINNER : $BEST_PARAMS"
    echo "  Time : ${BEST_TIME_MS} ms"
else
    echo "No valid configurations found."
fi

rm -f "$EXE"