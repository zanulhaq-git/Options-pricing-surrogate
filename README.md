Uses a neural network to approximate options prices and Greeks from an initial model.  Currently built on the Black-Scholes PDE, and validated on the closed-form solutions.

## Technologies used:
- Python language
- NumPy
- Torch
- Matplotlib

## Example results
![Analytical delta vs neural network delta]
https://github.com/zanulhaq-git/Options-pricing-surrogate/blob/master/delta_vs_nn.png

## Future updates:
- Bug fixes
- Adding compatiblity to other pricing models, including models without closed-form solutions
