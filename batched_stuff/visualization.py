import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pickle


if __name__ == "__main__":
    with open("results.pickle", "rb") as f:
        results = pickle.load(f)

    SKIP_METRICS = ["selectivity", "infidelity", "region_perturbation"]

    fig, axs = plt.subplots(1, len(results), figsize=(len(results) * 4, 5))
    runtime_ratios = []
    for i, (metric_name, result) in enumerate(results.items()):
        could_load = False
        try:
            batched_results = np.stack(result["scores"]["batched"], axis=1).astype(float)
            unbatched_results = np.stack(result["scores"]["unbatched"], axis=1).astype(float)
            could_load = True
        except:
            if isinstance(result["scores"]["batched"][0], dict):
                batched_results = np.array([sum(d.values()) for d in result["scores"]["batched"]])[:, None].astype(
                    float
                )
                unbatched_results = np.array([sum(d.values()) for d in result["scores"]["unbatched"]])[:, None].astype(
                    float
                )
                could_load = True
            pass

        # Plot runtimes
        batched_times = np.array(result["times"]["batched"])  # [np.array(result["times"]["batched"]) > 0]
        unbatched_times = np.array(result["times"]["unbatched"])  # [np.array(result["times"]["unbatched"]) > 0]
        batched_runtime, unbatched_runtime = (
            batched_times.mean(),
            unbatched_times.mean(),
        )
        axs[i].boxplot([batched_times, unbatched_times])  # , showfliers=False)
        axs[i].set_title(f"{metric_name} Spd-up: {round(unbatched_runtime / batched_runtime, 1)}x")
        axs[i].set_xticklabels(
            [
                f"batched {round(batched_runtime, 3)}s",
                f"unbatched {round(unbatched_runtime, 3)}s",
            ]
        )
        runtime_ratios.append(unbatched_runtime / batched_runtime)

        # skip metrics with major changes (since their results are definitively different)
        if metric_name in SKIP_METRICS:
            print(f"skipping {metric_name}")
            continue

        if not could_load:
            print(f"Could not load {metric_name}")
            continue
        # Check if all equal (close)
        if np.allclose(np.nan_to_num(batched_results), np.nan_to_num(unbatched_results), atol=1e-5):
            print(f"{metric_name} is VALID (all close)")
            continue

        # Check t-test otherwise at p < 0.05
        p_values = ttest_ind(batched_results, unbatched_results, axis=1).pvalue
        # If both implementations returned NaN, treat it as valid
        p_values[np.isnan(batched_results.mean(axis=1)) & np.isnan(unbatched_results.mean(axis=1))] = np.inf
        if np.all(p_values > 0.05):
            print(f"{metric_name} is VALID (t-test)")
        else:
            print(f"{metric_name} is INVALID (t-test) (p > 0.05 elements: {round((p_values > 0.05).mean() * 100, 2)}%)")
            print("p-values\n", p_values)
    fig.suptitle(f"Average speed-up: {round(sum(runtime_ratios) / len(runtime_ratios), 3)}x")
    plt.show()
