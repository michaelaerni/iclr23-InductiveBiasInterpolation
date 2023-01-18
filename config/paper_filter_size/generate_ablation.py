NOISE_FRACTION = (0.0, 0.2)
NUM_SEEDS = 5
NUM_INNER_SEEDS = 15
WIDTHS = (256, 512)
PEAK_LR = 0.13
FILTER_SIZES = (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 31)

if __name__ == "__main__":
    with open("synthetic_ablation_bindings.txt", "w") as f:
        for width in WIDTHS:
            for seed in range(1, NUM_SEEDS + 1):
                for training_seed in range(1, NUM_INNER_SEEDS + 1):
                    for noise_fraction in NOISE_FRACTION:
                        for filter_size in FILTER_SIZES:
                            bindings_str = \
                                f"width={width}" \
                                f" lr_schedule_peak_value={PEAK_LR}" \
                                f" noise_fraction={noise_fraction}" \
                                f" seed={seed}" \
                                f" training_seed={training_seed}" \
                                f" filter_size={filter_size}"

                            f.write(bindings_str + "\n")
