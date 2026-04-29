pymc.Data containers. These containers make it extremely easy to work with data in a PyMC model. They offer a range of
benefits, including:

Visualization of data as a component of your probabilistic graph

Access to labeled dimensions for readability and accessibility

Support for swapping out data for out-of-sample prediction, interpolation/extrapolation, forecasting, etc.

All data will be stored in your arviz.InferenceData, which is useful for plotting and reproducible workflows.

`with pm.Model() as no_data_model:
    x_data = pm.Data("x_data", x)
    y_data = pm.Data("y_data", y)
    beta = pm.Normal("beta")
    mu = pm.Deterministic("mu", beta * x_data)
    sigma = pm.Exponential("sigma", 1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)
    idata = pm.sample(random_seed=RANDOM_SEED)`
