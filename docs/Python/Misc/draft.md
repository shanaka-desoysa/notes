---
title: Draft 
date: 2024-12-09
author: Shanaka DeSoysa
description: Draft  WIP.
---

# Numba
Numba JIT Compilation:

The compute_percentiles function is compiled with Numba for faster percentile calculations.
The bootstrap_sampling function is also compiled with Numba to efficiently generate bootstrap indices.
Vectorized Sampling:

Bootstrap indices are generated in a single step using np.random.randint and applied to the DataFrame.
Preallocated Arrays:

Arrays for bootstrap_weights, bootstrap_random_variable, and bootstrap_comparison are preallocated for better memory management.
DataFrame to NumPy Conversion:

The DataFrame is converted to a NumPy array (df.values) for faster indexing during bootstrap sampling.
This refactored code should significantly improve performance, especially for large datasets and a high number of bootstrap


```python
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def compute_percentiles(data, alpha):
    """Compute lower and upper percentiles."""
    lower = np.percentile(data, alpha * 100 / 2, axis=0)
    upper = np.percentile(data, 100 - alpha * 100 / 2, axis=0)
    return lower, upper

@jit(nopython=True)
def bootstrap_sampling(df_values, num_bootstrap, num_samples):
    """Perform bootstrap sampling."""
    bootstrap_indices = np.random.randint(0, num_samples, size=(num_bootstrap, num_samples))
    return bootstrap_indices

def optimized_bootstrap_relative_weights(df, outcome, drivers, focal=None, num_bootstrap=10000, compare="No", alpha=0.05):
    def compute_rel_wts(sample):
        return relativeImp(sample, outcome, drivers)['rawRelaImpt'].values

    def random_variable(sample):
        return add_random_variable(sample, outcome, drivers)['rawRelaImpt'].values

    def compare_preds(sample):
        return compare_predictors(sample, outcome, drivers, focal)['weightDiff'].values

    # Convert DataFrame to NumPy array for faster processing
    df_values = df.values
    num_samples = len(df)

    # Preallocate arrays for bootstrap results
    num_drivers = len(drivers)
    bootstrap_weights = np.zeros((num_bootstrap, num_drivers))
    bootstrap_random_variable = np.zeros((num_bootstrap, num_drivers))
    bootstrap_comparison = np.zeros((num_bootstrap, num_drivers)) if compare == "Yes" else None

    # Generate bootstrap indices using Numba
    bootstrap_indices = bootstrap_sampling(df_values, num_bootstrap, num_samples)

    for i in range(num_bootstrap):
        sample = df.iloc[bootstrap_indices[i]]

        try:
            bootstrap_weights[i] = compute_rel_wts(sample)
            bootstrap_random_variable[i] = random_variable(sample)

            if compare == "Yes":
                bootstrap_comparison[i] = compare_preds(sample)
        except Exception as e:
            print(f"Error in bootstrap sample {i}: {e}")

    # Compute confidence intervals using Numba
    ci_relative_weights_lower, ci_relative_weights_upper = compute_percentiles(bootstrap_weights, alpha)
    ci_random_variable_lower, ci_random_variable_upper = compute_percentiles(bootstrap_random_variable, alpha)

    result = {
        'relative_weights': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_relative_weights_lower,
            'ci_upper': ci_relative_weights_upper,
            'ci_median': np.median(bootstrap_weights, axis=0),
        },
        'random_variable_diff': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_random_variable_lower,
            'ci_upper': ci_random_variable_upper,
            'ci_median': np.median(bootstrap_random_variable, axis=0),
        }
    }

    if compare == "Yes":
        ci_comparison_lower, ci_comparison_upper = compute_percentiles(bootstrap_comparison, alpha)
        comparisons = [f"{d}-{focal}" for d in drivers if d != focal]
        result['comparison_diff'] = {
            'comparison': comparisons,
            'ci_lower': ci_comparison_lower,
            'ci_upper': ci_comparison_upper,
            'ci_median': np.median(bootstrap_comparison, axis=0),
        }

    # Plot histograms
    plt.figure(figsize=(10, 4 * num_drivers))
    for i, driver in enumerate(drivers):
        weights = bootstrap_weights[:, i]
        median = np.median(weights)
        plt.subplot(num_drivers, 1, i + 1)
        plt.hist(weights, bins=50, alpha=0.5, color='blue', label='Relative Weights')
        plt.axvline(ci_relative_weights_lower[i], color='red', linestyle='--', label='Lower Bound')
        plt.axvline(ci_relative_weights_upper[i], color='green', linestyle='--', label='Upper Bound')
        plt.axvline(median, color='purple', linestyle='-', label='Median')
        plt.title(f"Distribution of Relative Weights for {driver}")
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

    return result
```

## Parallel Processing with Numba
```python
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

@jit(nopython=True)
def compute_percentiles(data, alpha):
    """Compute lower and upper percentiles manually."""
    num_samples, num_features = data.shape
    lower = np.zeros(num_features)
    upper = np.zeros(num_features)

    for j in range(num_features):
        sorted_column = np.sort(data[:, j])  # Sort each column manually
        lower_idx = int(alpha * 0.5 * num_samples)
        upper_idx = int((1 - alpha * 0.5) * num_samples)
        lower[j] = sorted_column[lower_idx]
        upper[j] = sorted_column[upper_idx]

    return lower, upper

@jit(nopython=True, parallel=True)
def bootstrap_sampling(data, num_bootstrap):
    """Perform bootstrap sampling and compute results."""
    num_samples, num_features = data.shape
    bootstrap_weights = np.zeros((num_bootstrap, num_features))
    for i in prange(num_bootstrap):
        indices = np.random.randint(0, num_samples, size=num_samples)
        sample = data[indices]
        # Compute mean manually along axis 0
        for j in range(num_features):
            bootstrap_weights[i, j] = sample[:, j].sum() / num_samples
    return bootstrap_weights

def optimized_bootstrap_relative_weights(df, outcome, drivers, num_bootstrap=10000, alpha=0.05):
    # Convert DataFrame to NumPy array (ensure uniform data type)
    data = df[drivers].to_numpy(dtype=np.float64)

    # Perform bootstrap sampling using Numba
    bootstrap_weights = bootstrap_sampling(data, num_bootstrap)

    # Compute confidence intervals
    ci_lower, ci_upper = compute_percentiles(bootstrap_weights, alpha)
    ci_median = np.median(bootstrap_weights, axis=0)

    # Prepare result
    result = {
        'relative_weights': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_median': ci_median,
        }
    }

    # Plot histograms
    num_drivers = len(drivers)
    plt.figure(figsize=(10, 4 * num_drivers))
    for i, driver in enumerate(drivers):
        weights = bootstrap_weights[:, i]
        median = ci_median[i]
        plt.subplot(num_drivers, 1, i + 1)
        plt.hist(weights, bins=50, alpha=0.5, color='blue', label='Relative Weights')
        plt.axvline(ci_lower[i], color='red', linestyle='--', label='Lower Bound')
        plt.axvline(ci_upper[i], color='green', linestyle='--', label='Upper Bound')
        plt.axvline(median, color='purple', linestyle='-', label='Median')
        plt.title(f"Distribution of Relative Weights for {driver}")
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

    return result
```



# Parallel Processing:

The ProcessPoolExecutor is used to distribute the bootstrap sampling across multiple CPU cores.
Each bootstrap iteration is handled by the bootstrap_worker function, which processes a single sample.
Efficient Data Handling:

Results from each worker are collected and converted into NumPy arrays for further processing.
Error Handling:

Errors in individual bootstrap samples are logged, and invalid results are skipped.
Confidence Interval Calculation:

Percentiles are calculated on the aggregated results after all workers complete their tasks.
This approach leverages parallelism to significantly reduce the runtime for large datasets and a high number of bootstrap iterations.

```python
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def compute_rel_wts(sample, outcome, drivers):
    return relativeImp(sample, outcome, drivers)['rawRelaImpt'].values

def random_variable(sample, outcome, drivers):
    return add_random_variable(sample, outcome, drivers)['rawRelaImpt'].values

def compare_preds(sample, outcome, drivers, focal):
    return compare_predictors(sample, outcome, drivers, focal)['weightDiff'].values

def bootstrap_worker(df, outcome, drivers, focal, compare):
    """Worker function to process a single bootstrap sample."""
    indices = np.random.choice(df.index, size=len(df), replace=True)
    sample = df.loc[indices]

    try:
        relative_weights = compute_rel_wts(sample, outcome, drivers)
        rand_weights = random_variable(sample, outcome, drivers)
        comp_weights = compare_preds(sample, outcome, drivers, focal) if compare == "Yes" else None
        return relative_weights, rand_weights, comp_weights
    except Exception as e:
        print(f"Error in bootstrap sample: {e}")
        return None, None, None

def parallel_bootstrap_relative_weights(df, outcome, drivers, focal=None, num_bootstrap=10000, compare="No", alpha=0.05):
    num_drivers = len(drivers)

    # Preallocate arrays for bootstrap results
    bootstrap_weights = []
    bootstrap_random_variable = []
    bootstrap_comparison = [] if compare == "Yes" else None

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(bootstrap_worker, df, outcome, drivers, focal, compare)
            for _ in range(num_bootstrap)
        ]

        for future in futures:
            relative_weights, rand_weights, comp_weights = future.result()
            if relative_weights is not None:
                bootstrap_weights.append(relative_weights)
                bootstrap_random_variable.append(rand_weights)
                if compare == "Yes":
                    bootstrap_comparison.append(comp_weights)

    # Convert results to NumPy arrays
    bootstrap_weights = np.array(bootstrap_weights)
    bootstrap_random_variable = np.array(bootstrap_random_variable)
    if compare == "Yes":
        bootstrap_comparison = np.array(bootstrap_comparison)

    # Compute confidence intervals
    ci_relative_weights = np.percentile(bootstrap_weights, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
    ci_random_variable = np.percentile(bootstrap_random_variable, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)

    result = {
        'relative_weights': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_relative_weights[0],
            'ci_upper': ci_relative_weights[1],
            'ci_median': np.median(bootstrap_weights, axis=0),
        },
        'random_variable_diff': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_random_variable[0],
            'ci_upper': ci_random_variable[1],
            'ci_median': np.median(bootstrap_random_variable, axis=0),
        }
    }

    if compare == "Yes":
        ci_comparison = np.percentile(bootstrap_comparison, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
        comparisons = [f"{d}-{focal}" for d in drivers if d != focal]
        result['comparison_diff'] = {
            'comparison': comparisons,
            'ci_lower': ci_comparison[0],
            'ci_upper': ci_comparison[1],
            'ci_median': np.median(bootstrap_comparison, axis=0),
        }

    # Plot histograms
    plt.figure(figsize=(10, 4 * num_drivers))
    for i, driver in enumerate(drivers):
        weights = bootstrap_weights[:, i]
        median = np.median(weights)
        plt.subplot(num_drivers, 1, i + 1)
        plt.hist(weights, bins=50, alpha=0.5, color='blue', label='Relative Weights')
        plt.axvline(ci_relative_weights[0][i], color='red', linestyle='--', label='Lower Bound')
        plt.axvline(ci_relative_weights[1][i], color='green', linestyle='--', label='Upper Bound')
        plt.axvline(median, color='purple', linestyle='-', label='Median')
        plt.title(f"Distribution of Relative Weights for {driver}")
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

    return result
```

## new

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, prange
from multiprocessing import Pool

@jit(nopython=True)
def compute_rel_wts(sample, outcome, drivers):
    return relativeImp(sample, outcome, drivers)['rawRelaImpt'].values

@jit(nopython=True)
def random_variable(sample, outcome, drivers):
    return add_random_variable(sample, outcome, drivers)['rawRelaImpt'].values

@jit(nopython=True)
def compare_preds(sample, outcome, drivers, focal):
    return compare_predictors(sample, outcome, drivers, focal)['weightDiff'].values

def bootstrap_sample(df, outcome, drivers, focal, compare):
    indices = np.random.choice(df.index, size=len(df), replace=True)
    sample = df.loc[indices]

    try:
        relative_weights = compute_rel_wts(sample, outcome, drivers)
        rand_weights = random_variable(sample, outcome, drivers)
        comp_weights = compare_preds(sample, outcome, drivers, focal) if compare == "Yes" else None
        return relative_weights, rand_weights, comp_weights
    except Exception as e:
        print(f"Error in bootstrap sample: {e}")
        return None, None, None

def bootstrap_relative_weights(df, outcome, drivers, focal=None, num_bootstrap=10000, compare="No", alpha=0.05):
    with Pool() as pool:
        results = pool.starmap(bootstrap_sample, [(df, outcome, drivers, focal, compare) for _ in range(num_bootstrap)])

    bootstrap_weights = [res[0] for res in results if res[0] is not None]
    bootstrap_random_variable = [res[1] for res in results if res[1] is not None]
    bootstrap_comparision = [res[2] for res in results if res[2] is not None]

    ci_relative_weights = np.percentile(bootstrap_weights, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
    ci_random_variable = np.percentile(bootstrap_random_variable, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)

    result = {
        'relative_weights': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_relative_weights[0],
            'ci_upper': ci_relative_weights[1],
            'ci_median': np.median(bootstrap_weights, axis=0),
        },
        'random_variable_diff': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_random_variable[0],
            'ci_upper': ci_random_variable[1],
            'ci_median': np.median(bootstrap_random_variable, axis=0)
        }
    }

    if compare == "Yes":
        ci_comparision = np.percentile(bootstrap_comparision, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
        comparisions = [f"{d}-{focal}" for d in drivers if d != focal]
        result['comparision_diff'] = {
            'comparision': comparisions,
            'ci_lower': ci_comparision[0],
            'ci_upper': ci_comparision[1],
            'ci_median': np.median(bootstrap_comparision)
        }

    num_drivers = len(drivers)
    plt.figure(figsize=(10, 4 * num_drivers))

    for i, driver in enumerate(drivers):
        weights = [rw[i] for rw in bootstrap_weights]
        median = np.median(weights)
        plt.subplot(num_drivers, 1, i + 1)
        plt.hist(weights, bins=50, alpha=0.5, color='blue', label='Relative Weights')
        plt.axvline(ci_relative_weights[0][i], color='red', linestyle='--', label='Lower Bound')
        plt.axvline(ci_relative_weights[1][i], color='green', linestyle='--', label='Upper Bound')
        plt.axvline(median, color='purple', linestyle='-', label='Median')
        plt.title(f"Distribution of Relative Weights for {driver}")
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

    return result

```

# Numba New
```python
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def compute_rel_wts_jit(sample, outcome_idx, driver_indices):
    # Placeholder for relativeImp logic
    # Replace with the actual implementation
    weights = np.zeros(len(driver_indices))
    for i, driver_idx in enumerate(driver_indices):
        weights[i] = np.mean(sample[:, driver_idx] * sample[:, outcome_idx])  # Example logic
    return weights

@njit
def random_variable_jit(sample, outcome_idx, driver_indices):
    # Placeholder for add_random_variable logic
    # Replace with the actual implementation
    weights = np.zeros(len(driver_indices))
    for i, driver_idx in enumerate(driver_indices):
        weights[i] = np.mean(sample[:, driver_idx] * sample[:, outcome_idx])  # Example logic
    return weights

@njit
def bootstrap_loop(data, outcome_idx, driver_indices, num_bootstrap, compare):
    n_samples = data.shape[0]
    n_drivers = len(driver_indices)
    bootstrap_weights = np.zeros((num_bootstrap, n_drivers))
    bootstrap_random_variable = np.zeros((num_bootstrap, n_drivers))
    bootstrap_comparision = np.zeros((num_bootstrap, n_drivers)) if compare else None

    for b in range(num_bootstrap):
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample = data[sample_indices]

        bootstrap_weights[b] = compute_rel_wts_jit(sample, outcome_idx, driver_indices)
        bootstrap_random_variable[b] = random_variable_jit(sample, outcome_idx, driver_indices)

        if compare:
            # Placeholder for compare_predictors logic
            bootstrap_comparision[b] = np.zeros(n_drivers)  # Replace with actual comparison logic

    return bootstrap_weights, bootstrap_random_variable, bootstrap_comparision

def bootstarp_relative_weights(df, outcome, drivers, focal=None, num_bootstrap=10000, compare="No", alpha=0.05):
    data = df.to_numpy()
    outcome_idx = df.columns.get_loc(outcome)
    driver_indices = [df.columns.get_loc(driver) for driver in drivers]
    compare_flag = compare == "Yes"

    # Run the bootstrap loop with Numba
    bootstrap_weights, bootstrap_random_variable, bootstrap_comparision = bootstrap_loop(
        data, outcome_idx, driver_indices, num_bootstrap, compare_flag
    )

    ci_level = (1 - alpha) * 100
    ci_relative_weights = np.percentile(bootstrap_weights, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
    ci_random_variable = np.percentile(bootstrap_random_variable, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)

    result = {
        'relative_weights': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_relative_weights[0],
            'ci_upper': ci_relative_weights[1],
            'ci_median': np.median(bootstrap_weights, axis=0),
        },
        'random_variable_diff': {
            'Outcome': outcome,
            'Drivers': drivers,
            'ci_lower': ci_random_variable[0],
            'ci_upper': ci_random_variable[1],
            'ci_median': np.median(bootstrap_random_variable, axis=0)
        }
    }

    if compare_flag:
        ci_comparision = np.percentile(bootstrap_comparision, [alpha * 100 / 2, 100 - alpha * 100 / 2], axis=0)
        comparisions = [f"{d}-{focal}" for d in drivers if d != focal]
        result['comparision_diff'] = {
            'comparision': comparisions,
            'ci_lower': ci_comparision[0],
            'ci_upper': ci_comparision[1],
            'ci_median': np.median(bootstrap_comparision, axis=0)
        }

    num_drivers = len(drivers)
    plt.figure(figsize=(10, 4 * num_drivers))

    for i, driver in enumerate(drivers):
        weights = bootstrap_weights[:, i]
        median = np.median(weights)
        plt.subplot(num_drivers, 1, i + 1)
        plt.hist(weights, bins=50, alpha=0.5, color='blue', label='Relative Weights')
        plt.axvline(ci_relative_weights[0][i], color='red', linestyle='--', label='Lower Bound')
        plt.axvline(ci_relative_weights[1][i], color='green', linestyle='--', label='Upper Bound')
        plt.axvline(median, color='purple', linestyle='-', label='Median')
        plt.title(f"Distribution of Relative Weights for {driver}")
        plt.ylabel('Frequency')
        plt.legend()
    plt.show()

    return result
```