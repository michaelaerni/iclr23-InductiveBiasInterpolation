NOISE_FRACTION = (0.0, 0.2)
NUM_SEEDS = 1
NUM_INNER_SEEDS = 5
NUM_TRAIN_ROTATIONS = (1, 2, 4, 12)

if __name__ == "__main__":
    with open("eurosat_ablation_bindings.txt", "w") as f:
        for seed in range(1, NUM_SEEDS + 1):
            for training_seed in range(1, NUM_INNER_SEEDS + 1):
                for noise_fraction in NOISE_FRACTION:
                    for num_train_rotations in NUM_TRAIN_ROTATIONS:
                        bindings_str = \
                            f"--bindings" \
                            f" noise_fraction={noise_fraction}" \
                            f" num_train_rotations={num_train_rotations}" \
                            f" seed={seed}" \
                            f" training_seed={training_seed}"

                        f.write(bindings_str + "\n")
