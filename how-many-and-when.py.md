```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

az.style.use("arviz-darkgrid")
```

## Change these constants to personalize your results

```python
NROF_SAMPLES = 1000  # ... to take for probability distributions
NROF_SAMPLE_LINES = 50  # ... to plot for probability distributions
NROF_DAYS = 400  # ... to calculate the expected number of stories for
NROF_STORIES = 100  # ... to calculate the expected number of days for
QUANTILES = np.array([5, 10, 20, 30, 50])  # ... to show the probability of
```

## Historical data

```python
df = pd.read_csv('data.csv', usecols=['Done'], parse_dates=['Done']).dropna()
df = df.rename(columns={"Done": "Date"}).set_index("Date").assign(Stories=1)
df = df.resample("D").count()
df.head()
```

```python
def plot_burn_up():
    _, ax = plt.subplots()
    ax.plot(df.index, df.cumsum().Stories)
    ax.set_title("Burn-up")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stories")
    return ax
```

```python
ax = plot_burn_up()
```

## The model

```python
with pm.Model() as m:
    pm.Data("count_data", df.Stories)
    pm.Lognormal("mu", mu=0, sigma=1)
    pm.Poisson("count", mu=m.mu, observed=m.count_data)
```

## Evaluate the model

What would the model expect based on the priors?

```python
with m:
    prior_pc = pm.sample_prior_predictive(NROF_SAMPLES)
```

```python
def plot_stories_per_day(*, prior=None, posterior_predictive=None, group="posterior"):
    ax = az.plot_ppc(az.from_pymc3(prior=prior_pc, model=m), group="prior", num_pp_samples=NROF_SAMPLE_LINES)
    ax.set_xlabel("Stories per Day")
    ax.set_ylabel("Probability")
    return ax
```

```python
ax = plot_stories_per_day(prior=prior_pc, group="prior")
ax.set_title("Expected stories per day before seeing any data")
None
```

```python
def plot_burn_up_ppc(ppc):
    ax = plot_burn_up()
    ax.plot(df.index, ppc["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
    return ax
```

```python
ax = plot_burn_up_ppc(prior_pc)
ax.set_title("Expected burn-up before seeing any data")
None
```

## Inferencing

Update the priors based on the data we've seen.

```python
with m:
    idata = pm.sample(return_inferencedata=True)
```

## Visualize posterior

Compare the trained model with the observed data.

```python
with m:
    post_pc = pm.sample_posterior_predictive(idata, NROF_SAMPLES)
```

```python
ax = plot_stories_per_day(prior=post_pc)
ax.set_title("Expected stories per day after seeing any data")
None
```

```python
ax = plot_burn_up_ppc(post_pc)
ax.set_title("Expected burn-up after seeing the data")
None
```

## Predictions

```python
with m:
    pm.set_data({
        "count_data": np.zeros(NROF_DAYS, dtype=int),  # generate data for the 100 days to come
    })
    post_pred = pm.sample_posterior_predictive(idata, NROF_SAMPLES)
```

```python
def plot_expected_burn_up():
    _, ax = plt.subplots()
    ax.plot(np.arange(NROF_DAYS),
            post_pred["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0),
            color="black", alpha=.1)
    ax.set_title("Expected burn-up for future stories")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stories")
    return ax
```

```python
ax = plot_expected_burn_up()
```

## How many stories will be done in X days?

```python
ax = plot_expected_burn_up()
ax.axvline(NROF_DAYS)
None
```

```python
stories_samples = post_pred["count"].cumsum(axis=1)[:,NROF_DAYS-1]  # of how many stories done
```

```python
def plot_prediction(samples):
    ax = az.plot_kde(samples, quantiles=QUANTILES/100)
    ax.set_ylabel("Probability")
    return ax
```

```python
ax = plot_prediction(stories_samples)
ax.set_title("Predicted number of stories")
ax.set_xlabel("Stories")
None
```

```python
def get_quantiles(name, samples):
    return pd.DataFrame({name: np.percentile(samples, QUANTILES).astype(int)},
                        index=[f"{100 - q}%" for q in QUANTILES])
```

```python
get_quantiles("stories", stories_samples)
```

## How many days will be needed for X stories?

```python
ax = plot_expected_burn_up()
ax.axhline(NROF_STORIES)
None
```

```python
days_samples = (post_pred["count"].cumsum(axis=1) < NROF_STORIES).sum(axis=1) + 1  # of nrof days until the stories are done
assert np.all(days_samples <= NROF_DAYS), f"Some scenarios need more than {NROF_DAYS} days"
```

```python
ax = plot_prediction(days_samples)
ax.set_title("Predicted number of days")
ax.set_xlabel("Days")
None
```

```python
get_quantiles("days", days_samples)
```
