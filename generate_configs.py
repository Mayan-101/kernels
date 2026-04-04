def generate_valid_configs():
    BM_list = [32, 64, 128]
    BN_list = [32, 64, 128]
    BK_list = [8, 16, 32]
    TM_list = [8]          # kernel does 2x float4 loads → TM must be ≥ 8
    TN_list = [8]          # same for TN
    extra_cols_list = [ 4, 8]   # 0 = no padding; others break bank conflicts

    valid_configs = []

    for BM in BM_list:
        for BN in BN_list:
            for BK in BK_list:
                for TM in TM_list:
                    for TN in TN_list:
                        for extra_cols in extra_cols_list:

                            #  1. tile divisibility 
                            if BM % TM != 0 or BN % TN != 0:
                                continue

                            #  2. actual thread count (kernel formula) 
                            threads = (BM // TM) * (BN // TN)
                            if threads < 32 or threads > 1024:
                                continue

                            #  3. BK must be float4-friendly 
                            if BK % 4 != 0:
                                continue

                            #  4. stride_A — must divide BM evenly 
                            #    kernel: stride_A = threads / (BK/4)
                            bk_div4 = BK // 4
                            if threads % bk_div4 != 0:
                                continue
                            stride_A = threads // bk_div4
                            if stride_A == 0 or BM % stride_A != 0:
                                continue

                            #  5. stride_B — must divide BK evenly 
                            #    kernel: stride_B = threads / (BN/4)
                            if BN % 4 != 0:
                                continue
                            bn_div4 = BN // 4
                            if threads % bn_div4 != 0:
                                continue
                            stride_B = threads // bn_div4
                            if stride_B == 0 or BK % stride_B != 0:
                                continue

                            #  6. extra_cols alignment
                            row_stride = (BN + extra_cols)
                            # float4 reads in compute loop need 4-float alignment
                            if row_stride % 4 != 0:
                                continue
                            # padding only useful when it actually breaks bank-conflicts
                            # (stride multiple of 32 = bad; skip pointless extra_cols)
                            if extra_cols != 0 and row_stride % 32 == 0:
                                continue

                            #  7. shared memory ≤ 48 KB 
                            smem_AT = BK * BM
                            smem_B  = BK * row_stride
                            smem_bytes = (smem_AT + smem_B) * 4
                            if smem_bytes > 49152:
                                continue

                            valid_configs.append(f"{BM} {BN} {BK} {TM} {TN} {extra_cols} {threads}")

    for cfg in valid_configs:
        print(cfg)

if __name__ == "__main__":
    generate_valid_configs()