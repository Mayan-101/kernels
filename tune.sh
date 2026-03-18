#!/usr/bin/env bash

# Configuration
SOURCE="kernel.cu"
EXE="./matmul_tune"
N=2048 

# Store best time as an INTEGER (microseconds) to avoid float comparison bugs
BEST_TIME_US=999999999
BEST_TIME_MS="9999.99"
BEST_PARAMS=""

echo "Starting Auto-Tuning..."
echo "BM, BN, BK, TM, TN | Time (ms)"
echo "--------------------------------"

for BM in 32 64 ; do
  for BN in 32 64; do
    for BK in 8 16 32; do
      for TM in 4 8; do
        for TN in 4 8; do
          
          # Calculate threads per block
          THREADS=$(( (BM / TM) * (BN / TN) ))
          
          # Skip invalid thread counts
          if [ $THREADS -gt 1024 ] || [ $THREADS -lt 32 ]; then continue; fi

          # Compile the kernel
          nvcc -O3 -D_BM=$BM -D_BN=$BN -D_BK=$BK -D_TM=$TM -D_TN=$TN \
               $SOURCE -o $EXE 2>/dev/null

          # If compilation fails, skip
          if [ $? -ne 0 ]; then continue; fi

          # 🚀 THE FIX: Aggressively filter output to ONLY grab the decimal number
          RAW_RESULT=$($EXE $N 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?' | head -n 1)
          
          # Skip if empty or zero (launch failed)
          if [ -z "$RAW_RESULT" ] || [ "$RAW_RESULT" == "0" ]; then 
              continue 
          fi

          echo "$BM, $BN, $BK, $TM, $TN | $RAW_RESULT ms"

          # Convert float milliseconds to integer microseconds (e.g., 40.545 -> 40545)
          RESULT_US=$(awk -v r="$RAW_RESULT" 'BEGIN { printf "%d", r * 1000 }')

          # Pure bash integer comparison (No more syntax errors!)
          if [ "$RESULT_US" -lt "$BEST_TIME_US" ] && [ "$RESULT_US" -gt 0 ]; then
            BEST_TIME_US=$RESULT_US
            BEST_TIME_MS=$RAW_RESULT
            BEST_PARAMS="BM=$BM, BN=$BN, BK=$BK, TM=$TM, TN=$TN"
          fi

        done
      done
    done
  done
done

echo "--------------------------------"
if [ -n "$BEST_PARAMS" ]; then
    echo "WINNER: $BEST_PARAMS with $BEST_TIME_MS ms"
else
    echo "No valid configurations found. Check your kernel.cu for runtime errors."
fi

# Clean up
rm -f $EXE