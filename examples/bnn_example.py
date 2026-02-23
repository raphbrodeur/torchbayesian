"""
    @file:              bnn_example.py
    @Author:            Raphael Brodeur

    @Creation Date:     10/2025
    @Last modification: 02/2026

    @Description:       This file contains an example of how to use 'torchbayesian' to make, train and use a BNN version
                        of any torch model.

                        It does not use 'torchbayesian' to do strict, proper stochastic variational inference, rather
                        this file serves as a tutorial for users wanting to quickly implement pragmatic uncertainty
                        quantification.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import torchbayesian.bnn as bnn


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ##################################
    # 1. Generate some training data #
    ##################################

    x_data = torch.linspace(-5, 5, 250).unsqueeze(1)
    y_data = torch.sin(1.5 * x_data) + 0.1 * torch.randn_like(x_data)

    dataset = TensorDataset(x_data, y_data)

    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True
    )


    ######################################
    # 2. Define any torch model as usual #
    ######################################

    # e.g. simple MLP with 4 hidden layers
    model = nn.Sequential(
        nn.Linear(1, 20), nn.ReLU(),    # Input -> Layer 1
        nn.Linear(20, 20), nn.ReLU(),   # Layer 1 -> Layer 2
        nn.Linear(20, 20), nn.ReLU(),   # Layer 2 -> Layer 3
        nn.Linear(20, 20), nn.ReLU(),   # Layer 3 -> Layer 4
        nn.Linear(20, 1)                # Layer 4 -> Output
    )


    ################################
    # 3. Wrap torch model as a BNN #
    ################################

    model = bnn.BayesianModule(model)

    # 'model' is now a BNN.

    # 'BayesianModule()' replaces the trainable parameters in the original model by variational posteriors whose
    # variational parameters are trainable. For a Gaussian variational posterior, this effectively doubles the number of
    # trainable parameters.

    # By default, 'BayesianModule()' sets a Gaussian variational posterior and a Gaussian prior N(0, 1) for every
    # original parameter. One could also give some other variational posterior or prior as argument, e.g. :
    #     model = BayesianModule(model, variational_posterior="pretrained", prior=("NORMAL", {"mu": 0., "sigma": 1.}))
    #
    # If one wants to use variational posteriors and priors that are not implemented in the package, simply register a
    # factory function for said custom posterior or prior to the factories 'PosteriorFactory' or 'PriorFactory', as
    # detailed in the docs of 'Factory' in file 'torchbayesian.bnn.utils.factories'. This will let 'BayesianModule()'
    # access the custom posterior or prior.

    model.to(device)    # Send bnn 'model' to device

    # It is recommended to only send the model to some device or dtype once the model has been made into a BNN.
    # If 'model' is sent to some device or dtype before being made into a BNN, then the new BNN variational parameters
    # are not by default on the appropriate device or dtype. This can be fixed by manually inputting the device and
    # dtype in the call to 'BayesianModule()' :
    #     model = bnn.BayesianModule(model, device=device, dtype=dtype)
    # or more simply by moving the parameters after the model has been made into a BNN :
    #     model = bnn.BayesianModule(model).to(device, dtype)


    #######################
    # 4. Training the BNN #
    #######################

    # Training works just as usual, but users can also use the KL divergence of the model to do proper variational
    # inference (VI)

    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return ((y_true - y_pred) ** 2).mean()

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-2
    )

    num_epochs = 500
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} in progress...")

        model.train()
        for batch in loader:
            x = batch[0].to(device)     # Get input
            y = batch[1].to(device)     # Get ground truth

            optimizer.zero_grad()

            # Monte Carlo estimation of the expected NLL.
            # Using more than one sample prediction and computing the average NLL reduces the variance of the gradient
            # estimation.
            #
            #     num_mc_samples = 5
            #     nll_accumulator = 0.
            #     for _ in range(num_mc_samples):
            #         y_pred = model(x)           # Sample a prediction from the model
            #
            #         nll_accumulator += mse(y, y_pred)
            #     nll = nll_accumulator / num_mc_samples
            #
            #     loss = nll + model.kl_divergence()
            #
            # While more noisy, a single sample prediction is often sufficient:

            y_pred = model(x)           # Sample a prediction from the model.

            # One can use an ELBO-like loss to follow VI more closely :
            #     loss = NLL(y, y_pred) + model.kl_divergence()
            # From a more pragmatic perspective, the KL divergence term can be thought of as a regularization term for
            # the BNN parameters.

            # In practice, many users prefer simpler heuristic forms, which correspond to proper VI only under some
            # assumptions.

            loss = mse(y, y_pred) + 1e-1 * model.kl_divergence(reduction="mean")    # Loss

            # Theoretically (and by default), KL divergence is summed over all layers, but using a mean reduction
            # is a useful heuristic to help the training of the model be more scalable (using 'reduction=mean', one can
            # change the number of parameters in the model without directly affecting the magnitude of the training
            # objective).

            # Another heuristic used in practice is to scale the KL divergence term by some small coefficient (1e-1 in
            # this case). This can be done as a trade-off in an attempt to improve data fit at the cost of less
            # regularization.

            # A scaling coefficient could also be used in some cases to account for mini-batches, e.g. in the case of a
            # loss summed over the examples in the mini-batch :
            #     loss = dice_loss(reduction="batch_sum") + batch_size * model.kl_divergence()

            loss.backward()
            optimizer.step()


    ######################################
    # 5. How to evaluate the trained BNN #
    ######################################

    test_domain = torch.linspace(-8.5, 8.5, 5000).unsqueeze(1)

    # For each forward pass, the BNN model samples new weights from the learned posterior distributions. Therefore, two
    # forward passes may output different values.


    # 5.1 Single forward pass :

    # One can still do a single forward pass, just like with a regular model, but this is effectively drawing a single
    # sample from the BNN/learned predictive distribution.

    # Single forward pass of the model
    model.eval()
    with torch.no_grad():
        y_pred = model(test_domain.to(device))  # Draws parameters and uses them for a single forward pass

    # Visualization
    fig, ax = plt.subplots()
    ax.plot(test_domain.squeeze(1).cpu(), y_pred.squeeze(1).cpu(), color="#4F609C", zorder=2, label="Prediction")
    ax.scatter(-100., -100., color="black", s=10, marker="o", zorder=1, label="Training data")  # Dummy point for legend
    for x, y in dataset:
        ax.scatter(x.item(), y.item(), color="black", s=10, marker="o", zorder=1)
    ax.set_title("Prediction of 1 sample from the BNN")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-3.0, 3.0)
    ax.legend(loc="upper left")
    plt.show()


    # 5.2 Multiple forward passes :

    # By drawing multiple samples from the BNN (which is done by doing multiple forward passes over the same input), we
    # can better characterize the underlying predictive distribution of the trained model and its variability.

    fig, ax = plt.subplots()

    # Multiple forward passes of the model
    num_bnn_samples = 1000
    for _ in range(num_bnn_samples):
        model.eval()
        with torch.no_grad():
            y_pred = model(test_domain.to(device))  # Draws parameters and uses them for a single forward pass

    # Visualization
        ax.plot(test_domain.squeeze(1).cpu(), y_pred.squeeze(1).cpu(), color="#4F609C", alpha=0.01, zorder=2)
    ax.scatter(-100., -100., color="black", s=10, marker="o", zorder=1, label="Training data")  # Dummy point for legend
    for x, y in dataset:
        ax.scatter(x.item(), y.item(), color="black", s=10, marker="o", zorder=1)
    ax.set_title(f"Predictions from {num_bnn_samples} samples from the BNN")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-3.0, 3.0)
    ax.legend(loc="upper left")
    plt.show()


    # 5.3 BNN prediction :

    # To make a prediction using the BNN model, one can do multiple forward passes of the model and get statistics from
    # them to characterize the predictive distribution and the model's uncertainty about some output.

    # Multiple forward passes of the model
    bnn_samples = []
    num_bnn_samples = 1000              # In practice, 20 should be enough for most applications
    for _ in range(num_bnn_samples):
        model.eval()
        with torch.no_grad():
            y_pred = model(test_domain.to(device))  # Draws parameters and uses them for a single forward pass
            bnn_samples.append(y_pred.squeeze().cpu().numpy())

    # Statistics over the BNN samples
    y_mean = np.mean(bnn_samples, axis=0)   # Mean prediction of all the samples
    y_std = np.std(bnn_samples, axis=0)     # Standard deviation of the predictions from all the samples

    # Visualization
    fig, ax = plt.subplots()
    ax.plot(test_domain.squeeze(1).cpu(), y_mean, color="#4F609C", zorder=2, label=f"Mean prediction")
    ax.fill_between(
        test_domain.squeeze(1).cpu(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="#C0DEF0",
        zorder=0,
        label="±2σ"
    )
    ax.scatter(-100., -100., color="black", s=10, marker="o", zorder=1, label="Training data")  # Dummy point for legend
    for x, y in dataset:
        ax.scatter(x.item(), y.item(), color="black", s=10, marker="o", zorder=1)
    ax.set_title(f"Prediction and uncertainty using {num_bnn_samples} BNN samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-3.0, 3.0)
    ax.legend(loc="upper left")
    plt.show()
