# Implied Volatiltity Surface Visualizer

This repository allows users to visualize 3D implied volatility surfaces of a given stock. This uses options and uses the mathematics behind options pricing to work backwards and derive its theoretical implied volatility.

## Contributors

- Hansel Chen

## Method

Using the Black-Scholes formula, we input an option's price, time to expiry, risk free rate, dividend yield, and strike price to work backwards to find the implied volatility. Since Black-Scholes is a transcendal equation. We used scipy to optimize the closest possible implied volatility following the equation.

Some limitations of this model is that this model assumes a fixed risk free rate and dividend yield, which is not the case in the real world. Black-Scholes also assumes the right for exercise only at expiration, but this is not the case for American options.

## Usage

Consult presentation.ipynb for the full rundown of our exploration into this strategy.
