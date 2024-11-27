import sys

sys.path.append("..")
import quantus
from quantus.helpers.model.models import LeNet
import torch
import torchvision
from torchvision import transforms
from captum.attr import Saliency, InputXGradient, IntegratedGradients
import time
import pickle

REPETITIONS = 1
"""METRIC_PAIRS = {
    "pixel_flipping": {
        "batched": quantus.BatchPixelFlipping(return_auc_per_sample=True),
        "unbatched": quantus.PixelFlipping(return_auc_per_sample=True),
    },
    "monotonicity": {
        "batched": quantus.BatchMonotonicity(),
        "unbatched": quantus.Monotonicity(),
    },
    "monotonicity_correlation": {
        "batched": quantus.BatchMonotonicityCorrelation(nr_samples=20),
        "unbatched": quantus.MonotonicityCorrelation(nr_samples=20),
    },
    "faithfulness_estimate": {
        "batched": quantus.BatchFaithfulnessEstimate(),
        "unbatched": quantus.FaithfulnessEstimate(),
    },
    "faithfulness_correlation": {
        "batched": quantus.BatchFaithfulnessCorrelation(return_aggregate=False),
        "unbatched": quantus.FaithfulnessCorrelation(return_aggregate=False),
    },
    "infidelity": {
        "batched": quantus.BatchInfidelity(),
        "unbatched": quantus.Infidelity(),
    },
    "irof": {
        "batched": quantus.BatchIROF(return_aggregate=False),
        "unbatched": quantus.IROF(return_aggregate=False),
    },
    "region_perturbation": {
        "batched": quantus.BatchRegionPerturbation(),
        "unbatched": quantus.RegionPerturbation(),
    },
    "road": {
        "batched": quantus.BatchROAD(),
        "unbatched": quantus.ROAD(),
    },
    "selectivity": {
        "batched": quantus.BatchSelectivity(),
        "unbatched": quantus.Selectivity(),
    },
    "sensitivity_n": {
        "batched": quantus.BatchSensitivityN(),
        "unbatched": quantus.SensitivityN(),
    },
    "sufficiency": {
        "batched": quantus.BatchSufficiency(),
        "unbatched": quantus.Sufficiency(),
    },
}
"""

METRIC_PAIRS = {
    "avg_sensitivity": {
        "batched": quantus.BatchAvgSensitivity(),
        "unbatched": quantus.AvgSensitivity(),
    },
    "consistency": {
        "batched": quantus.BatchConsistency(),
        "unbatched": quantus.Consistency(),
    },
    "continuity": {
        "batched": quantus.BatchContinuity(),
        "unbatched": quantus.Continuity(),
    },
    "local_lipschitz_estimate": {
        "batched": quantus.BatchLocalLipschitzEstimate(),
        "unbatched": quantus.LocalLipschitzEstimate(),
    },
    "max_sensitivity": {
        "batched": quantus.BatchMaxSensitivity(),
        "unbatched": quantus.MaxSensitivity(),
    },
    "relative_input_stability": {
        "batched": quantus.BatchRelativeInputStability(),
        "unbatched": quantus.RelativeInputStability(),
    },
    "relative_output_stability": {
        "batched": quantus.BatchRelativeOutputStability(),
        "unbatched": quantus.RelativeOutputStability(),
    },
    "relative_representation_stability": {
        "batched": quantus.BatchRelativeRepresentationStability(),
        "unbatched": quantus.RelativeRepresentationStability(),
    },
    "sparseness": {
        "batched": quantus.BatchSparseness(),
        "unbatched": quantus.Sparseness(),
    },
    "non_sensitivity": {
        "batched": quantus.BatchNonSensitivity(),
        "unbatched": quantus.NonSensitivity(),
    },
    "random_logit": {
        "batched": quantus.BatchRandomLogit(),
        "unbatched": quantus.RandomLogit(),
    },
}

METRIC_PAIRS = {
    "random_logit": {
        "batched": quantus.BatchRandomLogit(num_classes=10),
        "unbatched": quantus.RandomLogit(num_classes=10),
    },
}

if __name__ == "__main__":
    # Enable GPU.
    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
    model = LeNet()
    if device.type == "cpu":
        model.load_state_dict(torch.load("../tests/assets/mnist", map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load("../tests/assets/mnist"))
    model.eval()

    # Load datasets and make loaders.
    test_set = torchvision.datasets.MNIST(
        root="../sample_data",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Load a batch of inputs and outputs to use for XAI evaluation.
    for x_batch, y_batch in test_loader:
        break

    # Generate Integrated Gradients attributions of the first batch of the test set.
    a_batch_saliency = (
        Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).detach().cpu().numpy()
    )

    # Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
    x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

    results = {}
    for metric_name, metric_dict in METRIC_PAIRS.items():
        print(metric_name)
        repetition_results, repetition_times = {"batched": [], "unbatched": []}, {
            "batched": [],
            "unbatched": [],
        }
        for repetition_index in range(REPETITIONS):
            for metric_type, metric in metric_dict.items():
                start_time = time.monotonic()
                scores = metric(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch_saliency,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={"method": "Saliency"},
                )
                repetition_times[metric_type].append(time.monotonic() - start_time)
                repetition_results[metric_type].append(scores)

                print(metric_type, time.monotonic() - start_time, scores)

        # Store results and times
        results[metric_name] = {}
        results[metric_name]["scores"] = repetition_results
        results[metric_name]["times"] = repetition_times
        print()

    with open(f"results_robustness_complexity_randomisation.pickle", "wb") as f:
        pickle.dump(results, f)
