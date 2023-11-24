"""This module provides some functionality to evaluate different explanation methods on several evaluation criteria."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import warnings
from typing import Union, Callable, Dict, Optional, List

import numpy as np

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.explanation_func import explain


def evaluate(
    metrics: Dict,
    xai_methods: Union[Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]],
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    s_batch: Union[np.ndarray, None] = None,
    agg_func: Callable = lambda x: x,
    progress: bool = False,
    explain_func_kwargs: Optional[dict] = None,
    call_kwargs: Union[Dict, Dict[str, Dict]] = None,
    **kwargs,
) -> Optional[dict]:
    """
    Evaluate different explanation methods using specified metrics.

    Parameters
    ----------
    metrics : dict
        A dictionary of initialized evaluation metrics. See quantus.AVAILABLE_METRICS.
        Example: {'Robustness': quantus.MaxSensitivity(), 'Faithfulness': quantus.PixelFlipping()}

    xai_methods : dict
        A dictionary specifying the explanation methods to evaluate, which can be structured in three ways:

        1) Dict[str, Dict] for built-in Quantus methods:

            Example:
            xai_methods = {
                'IntegratedGradients': {
                    'n_steps': 10,
                    'xai_lib': 'captum'
                },
                'Saliency': {
                    'xai_lib': 'captum'
                }
            }

            - See quantus.AVAILABLE_XAI_METHODS_CAPTUM for supported captum methods.
            - See quantus.AVAILABLE_XAI_METHODS_TF for supported tensorflow methods.
            - See https://github.com/chr5tphr/zennit for supported zennit methods.
            - Read more about the explanation function arguments here:
              <https://quantus.readthedocs.io/en/latest/docs_api/quantus.functions.explanation_func.html#quantus.functions.explanation_func.explain>

        2) Dict[str, Callable] for custom methods:

            Example:
            xai_methods = {
                'custom_own_xai_method': custom_explain_function
            }

            - Here, you can provide your own callable that mirrors the input and outputs of the quantus.explain() method.

        3) Dict[str, np.ndarray] for pre-calculated attributions:

            Example:
            xai_methods = {
                'LIME': precomputed_numpy_lime_attributions,
                'GradientShap': precomputed_numpy_shap_attributions
            }

            - Note that some metrics, e.g., quantus.MaxSensitivity() within the robustness category,
              requires the explanation function to be passed (as this is used in the evaluation logic).

        It is also possible to pass a combination of the above.

    model: Union[torch.nn.Module, tf.keras.Model]
        A torch or tensorflow model that is subject to explanation.

    x_batch: np.ndarray
        A np.ndarray containing the input data to be explained.

    y_batch: np.ndarray
        A np.ndarray containing the output labels corresponding to x_batch.

    s_batch: np.ndarray, optional
        A np.ndarray containing segmentation masks that match the input.

    agg_func: Callable
        Indicates how to aggregate scores, e.g., pass np.mean.

    progress: bool
        Indicates if progress should be printed to standard output.

    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call. Pass None if using Dict[str, Dict] type for xai_methods.

    call_kwargs: Dict[str, Dict]
        Keyword arguments for the call of the metrics. Keys are names for argument sets, and values are argument dictionaries.

    kwargs: optional
        Deprecated keyword arguments for the call of the metrics.

    Returns
    -------
    results: dict
        A dictionary with the evaluation results.
    """

    warn.check_kwargs(kwargs)

    if xai_methods is None:
        print("Define the explanation methods that you want to evaluate.")
        return None

    if metrics is None:
        print(
            "Define the Quantus evaluation metrics that you want to evaluate the explanations against."
        )
        return None

    if call_kwargs is None:
        call_kwargs = {"call_kwargs_empty": {}}
    elif not isinstance(call_kwargs, Dict):
        raise TypeError("xai_methods type is not Dict[str, Dict].")

    results: Dict[str, dict] = {}
    explain_funcs: Dict[str, Callable] = {}

    if not isinstance(xai_methods, dict):
        "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."

    for method, value in xai_methods.items():

        results[method] = {}

        if callable(value):

            explain_funcs[method] = value
            explain_func = value
            assert (
                explain_func_kwargs is not None
            ), "Pass explain_func_kwargs as a dictionary."

            # Asserts.
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **{**explain_func_kwargs, **{"method": method}},
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, Dict):

            if explain_func_kwargs is not None:
                warnings.warn(
                    "Passed explain_func_kwargs will be ignored when passing type Dict[str, Dict] as xai_methods."
                    "Pass explanation arguments as dictionary values."
                )

            explain_func_kwargs = value
            explain_funcs[method] = explain

            # Generate explanations.
            a_batch = explain(
                model=model, inputs=x_batch, targets=y_batch, **explain_func_kwargs
            )
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)

            # Asserts.
            asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        elif isinstance(value, np.ndarray):
            explain_funcs[method] = explain
            a_batch = value

        else:

            raise TypeError(
                "xai_methods type is not in: Dict[str, Callable], Dict[str, Dict], Dict[str, np.ndarray]."
            )

        if explain_func_kwargs is None:
            explain_func_kwargs = {}

        for (metric, metric_func) in metrics.items():

            results[method][metric] = {}

            for (call_kwarg_str, call_kwarg) in call_kwargs.items():

                if progress:
                    print(
                        f"Evaluating {method} explanations on {metric} metric on set of call parameters {call_kwarg_str}..."
                    )

                results[method][metric][call_kwarg_str] = agg_func(
                    metric_func(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        s_batch=s_batch,
                        explain_func=explain_funcs[method],
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
                        **call_kwarg,
                        **kwargs,
                    )
                )

    return results
