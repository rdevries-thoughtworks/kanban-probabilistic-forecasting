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
plt.plot(df.index, df.cumsum().Stories.values)
plt.title("Burn-up")
plt.xlabel("Date")
plt.ylabel("Stories");
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
az.plot_ppc(az.from_pymc3(prior=prior_pc, model=m), group="prior", num_pp_samples=NROF_SAMPLE_LINES)
plt.title("Expected stories per day before seeing any data")
plt.xlabel("Stories per Day")
plt.ylabel("Probability");
```

```python
plt.plot(df.index, prior_pc["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
plt.plot(df.index, df.cumsum().Stories)
plt.title("Expected burn-up before seeing any data")
plt.xlabel("Date")
plt.ylabel("Stories");
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
az.plot_ppc(az.from_pymc3(posterior_predictive=post_pc, model=m), num_pp_samples=NROF_SAMPLE_LINES)
plt.title("Expected stories per day after seeing the data")
plt.xlabel("Stories per Day")
plt.ylabel("Probability");
```

```python
plt.plot(df.index, post_pc["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
plt.plot(df.index, df.cumsum().Stories)
plt.title("Expected burn-up after seeing the data")
plt.xlabel("Date")
plt.ylabel("Stories");
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
x = np.arange(NROF_DAYS)
plt.plot(x, post_pred["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
plt.title("Expected burn-up for future stories")
plt.xlabel("Days")
plt.ylabel("Stories");
```

## How many stories will be done in X days?

```python
plt.plot(x, post_pred["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
plt.axvline(NROF_DAYS)
plt.title("Expected burn-up for future stories")
plt.xlabel("Days")
plt.ylabel("Stories");
```

```python
samples = post_pred["count"].cumsum(axis=1)[:,NROF_DAYS-1]  # of how many stories done
```

```python
az.plot_kde(samples, quantiles=QUANTILES/100)
plt.title("Predicted number of stories")
plt.xlabel("Stories per Day")
plt.ylabel("Probability");
```

```python
pd.DataFrame({"stories": np.percentile(samples, QUANTILES).astype(int)}, index=[f"{100 - q}%" for q in QUANTILES])
```

## How many days will be needed for X stories?

```python
plt.plot(x, post_pred["count"][:NROF_SAMPLE_LINES,:].T.cumsum(axis=0), color="black", alpha=.1)
plt.axhline(NROF_STORIES);
```

```python
samples = (post_pred["count"].cumsum(axis=1) < NROF_STORIES).sum(axis=1) + 1  # of nrof days until the stories are done
# TODO: assert that all stories are done in NROF_DAYS, otherwise the predicted number of days will be off
```

```python
az.plot_kde(samples, quantiles=QUANTILES/100)
plt.title("Predicted number of days")
plt.xlabel("Days")
plt.ylabel("Probability");
```

```python
pd.DataFrame({"days": np.percentile(samples, QUANTILES)}, index=[f"{100 - q}%" for q in QUANTILES])
```
