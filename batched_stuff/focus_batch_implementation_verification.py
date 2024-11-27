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
import numpy as np
from torch.nn import functional as F

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
    "sufficiency": {
        "batched": quantus.BatchFocus(),
        "unbatched": quantus.Focus(),
    },
}

if __name__ == "__main__":
    # Enable GPU.
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

    # Load a batch of inputs and outputs to use for XAI evaluation.
    for x_batch, y_batch in test_loader:
        break

    x_mosaic_batch = []
    y_mosaic_batch = []
    for i in range(0, x_batch.shape[0], 4):
        x_mosaic = torch.zeros(1, 28 * 2, 28 * 2)
        x_mosaic[:, :28, :28] = x_batch[i]
        x_mosaic[:, :28, 28:] = x_batch[i + 1]
        x_mosaic[:, 28:, :28] = x_batch[i + 2]
        x_mosaic[:, 28:, 28:] = x_batch[i + 3]
        x_mosaic_batch.append(x_mosaic)
        y_mosaic_batch.append(y_batch[i])
    x_mosaic_batch = torch.stack(x_mosaic_batch, axis=0)
    x_mosaic_batch = F.interpolate(x_mosaic_batch, scale_factor=0.5)
    y_mosaic_batch = torch.tensor(y_mosaic_batch)

    p_mosaic_batch = [(1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1)]

    # Generate Integrated Gradients attributions of the first batch of the test set.
    a_batch_saliency = (
        Saliency(model)
        .attribute(inputs=x_mosaic_batch, target=y_mosaic_batch, abs=True)
        .sum(axis=1)
        .detach()
        .cpu()
        .numpy()
    )

    # Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
    x_mosaic_batch, y_mosaic_batch = x_mosaic_batch.cpu().numpy(), y_mosaic_batch.cpu().numpy()

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
                    x_batch=x_mosaic_batch,
                    y_batch=y_mosaic_batch,
                    a_batch=a_batch_saliency,
                    custom_batch=p_mosaic_batch,
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
