#!/usr/bin/env python3

def generate_valid_configs():
    # Base search space
    BM_list = [32, 64, 128]
    BN_list = [32, 64, 128]
    BK_list = [8, 16, 32]
    TM_list = [4, 8]
    TN_list = [4, 8]

    valid_configs = []

    for BM in BM_list:
        for BN in BN_list:
            for BK in BK_list:
                if BK % 4 != 0: continue
                
                for TM in TM_list:
                    for TN in TN_list:
                        # User constraint: BLOCK_SIZE is exactly BM
                        BLOCK_SIZE = BM 

                        # 1. Compute constraints
                        if BM % TM != 0 or BN % TN != 0: continue
                        
                        compute_threads = (BM // TM) * (BN // TN)
                        if compute_threads < 32 or compute_threads > 1024: continue

                        # 2. Vectorized load constraints
                        if BN % 4 != 0: continue
                        
                        bk_div4 = BK // 4
                        if BLOCK_SIZE % bk_div4 != 0: continue
                        stride_A = BLOCK_SIZE // bk_div4
                        
                        bn_div4 = BN // 4
                        if BLOCK_SIZE % bn_div4 != 0: continue
                        stride_B = BLOCK_SIZE // bn_div4

                        if stride_A == 0 or BM % stride_A != 0: continue
                        if stride_B == 0 or BK % stride_B != 0: continue

                        # 3. Shared memory constraint (Max 48KB)
                        smem_bytes = (BK * BM + BK * BN) * 4
                        if smem_bytes > 49152: continue

                        # If it passes everything, it's a valid config!
                        valid_configs.append(f"{BM} {BN} {BK} {TM} {TN} {BLOCK_SIZE} {compute_threads}")

    # Output to stdout so bash can read it
    for config in valid_configs:
        print(config)

if __name__ == "__main__":
    generate_valid_configs()