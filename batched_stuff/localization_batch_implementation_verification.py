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
        "batched": quantus.BatchMonotonicityCorrelation(nr_samples=10),
        "unbatched": quantus.MonotonicityCorrelation(nr_samples=10),
    },
    "faithfulness_estimate": {
        "batched": quantus.BatchFaithfulnessEstimate(),
        "unbatched": quantus.FaithfulnessEstimate(),
    },
    "faithfulness_correlation": {
        "batched": quantus.BatchFaithfulnessCorrelation(return_aggregate=False),
        "unbatched": quantus.FaithfulnessCorrelation(return_aggregate=False),
    },
}"""
METRIC_PAIRS = {
    "attribution_localisation": {
        "batched": quantus.BatchAttributionLocalisation(),
        "unbatched": quantus.AttributionLocalisation(),
    },
    "auc": {
        "batched": quantus.BatchAUC(),
        "unbatched": quantus.AUC(),
    },
    "pointing_game": {
        "batched": quantus.BatchPointingGame(),
        "unbatched": quantus.PointingGame(),
    },
    "relevance_mass_accuracy": {
        "batched": quantus.BatchRelevanceMassAccuracy(),
        "unbatched": quantus.RelevanceMassAccuracy(),
    },
    "relevance_rank_accuracy": {
        "batched": quantus.BatchRelevanceRankAccuracy(),
        "unbatched": quantus.RelevanceRankAccuracy(),
    },
    "top_k_intersection": {
        "batched": quantus.BatchTopKIntersection(),
        "unbatched": quantus.TopKIntersection(),
    },
}

if __name__ == "__main__":
    # Enable GPU.
    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
    # Load pre-trained ResNet18 model.
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()

    # Adjust this path.
    path_to_files = "../tutorials/assets/imagenet_samples"

    # Load test data and make loaders.
    x_batch = torch.load(f"{path_to_files}/x_batch.pt")
    y_batch = torch.load(f"{path_to_files}/y_batch.pt")
    s_batch = torch.load(f"{path_to_files}/s_batch.pt")
    x_batch, s_batch, y_batch = x_batch.to(device), s_batch.to(device), y_batch.to(device)

    # Generate Integrated Gradients attributions of the first batch of the test set.
    a_batch_saliency = (
        Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).detach().cpu().numpy()
    )

    # Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
    x_batch, y_batch, s_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy(), s_batch.cpu().numpy()[:, None]

    results = {}
    for metric_name, metric_dict in METRIC_PAIRS.items():
        repetition_results, repetition_times = {"batched": [], "unbatched": []}, {
            "batched": [],
            "unbatched": [],
        }
        for repetition_index in range(REPETITIONS):
            for metric_type, metric in metric_dict.items():
                start_time = time.time()
                scores = metric(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    s_batch=s_batch,
                    a_batch=a_batch_saliency,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={"method": "Saliency"},
                )
                repetition_times[metric_type].append(time.time() - start_time)
                repetition_results[metric_type].append(scores)

                print(metric_type, scores, time.time() - start_time)

        # Store results and times
        results[metric_name] = {}
        results[metric_name]["scores"] = repetition_results
        results[metric_name]["times"] = repetition_times

    with open(f"results.pickle", "wb") as f:
        pickle.dump(results, f)
