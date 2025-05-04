# Rowing Speed Prediction from Per-Stroke Telemetry Data
**Author:** Nate Ly  
**Date:** May 4, 2025

## Introduction

In competitive rowing, the goal is to travel a distance of 2000 meters as fast as possible. The fastest boat, called an Eight, has eight rowers and a coxswain. Boat speed comes from a combination of power output from the athletes and technical cohesion. While maximizing power output relies on training the human body to its physiological limits, technical cohesion depends on neuromuscular synchronization and practice. Although sports science has made great strides in optimizing human physiology through training and nutrition, rowing technique remains the bottleneck of boat speed.

Traditional fluid-dynamics models can predict boat motion, but they require extreme care in specifying boundary conditions and are computationally expensive. Moreover, they cannot capture human variability due to fatigue or subtle technical differences. A data-driven approach “priced in” these unmodeled factors, offering not only a cheaper but potentially more practical and insightful method.

**General Approach:**
- Collect per-stroke data through special oarlocks equipped with strain gauges.
- Clean and filter the data to retain only “quality” strokes—i.e., when all eight rowers are rowing with intent and the boat is neither spinning nor drilling.
- Train a neural network to map the extracted features to boat speed.

**Challenges:**
- The telemetry oarlocks are in beta testing, endure substantial and repetitive strain, and are calibrated by hand, which can introduce skew in the measurements.
- Determining the best method to filter for “quality” strokes.
- Tuning machine-learning hyperparameters to optimize loss and generalization.

## Data

The data were collected from Yale Heavyweight Crew on April 7, 2025, using telemetry from two Eights, with the written permission of all 16 rowers and the head coach. Each per-stroke record includes:
- Power (W)
- Rate (strokes per minute)
- Work (J)
- Blade angles: catch angle, slip, wash, finish angle, connected length (degrees)
- Peak force angle (degrees)
- Catch time, stroke duration (seconds)

The target variable, boat speed (m/s), was measured via GPS.

## Methodology

### Problem Formulation

Let each stroke be represented by a feature vector \( \mathbf{x} \in \mathbb{R}^d \) and its measured speed by \( y \in \mathbb{R} \). Introduce a fixed offset \( b_0 \in \mathbb{R} \) to account for baseline speed. We then learn a residual function:

$$
g: \mathbb{R}^d \rightarrow \mathbb{R}
$$

such that:

$$
g(\mathbf{x}) \approx y - b_0
$$

The full predictor is:

$$
\hat{y} = g(\mathbf{x}) + b_0
$$

### Model Architecture

We implement a feed‐forward neural network in PyTorch with four fully connected layers and ReLU activations. The layer dimensions are \( d \to 64 \to 32 \to 16 \to 1 \). Let \( \mathbf{h}^{(0)} = \mathbf{x} \). Then for \( i = 1, 2, 3 \):

$$
\begin{aligned}
\mathbf{z}^{(i)} &= w^{(i)} \cdot \mathbf{h}^{(i-1)} + \mathbf{b}^{(i)} \\
\mathbf{h}^{(i)} &= \mathrm{ReLU}(\mathbf{z}^{(i)})
\end{aligned}
$$

$$
\mathbf{z}^{(4)} = w^{(4)} \cdot \mathbf{h}^{(3)} + b^{(4)}, \quad \hat{y} = \mathbf{z}^{(4)}
$$

### Loss Function and Optimization

We aim to minimize the Mean Squared Error over \( N \) strokes:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N (f(\mathbf{x}_i) - y_i)^2
$$

Gradients are computed via backpropagation and weights are adjusted using PyTorch's Adam optimizer:

- \( g^{(t)} = \nabla_w \mathcal{L}^{(t)} = \frac{\partial \text{Loss}}{\partial \text{Weights}} \)
- \( m^{(t)} = \beta_1 \cdot m^{(t-1)} + (1 - \beta_1) \cdot g^{(t)} \)
- \( v^{(t)} = \beta_2 \cdot v^{(t-1)} + (1 - \beta_2) \cdot (g^{(t)})^2 \)

Bias-corrected:

$$
\hat{m}^{(t)} = \frac{m^{(t)}}{1 - \beta_1^t}, \quad \hat{v}^{(t)} = \frac{v^{(t)}}{1 - \beta_2^t}
$$

Weight update rule:

$$
w^{(t+1)} = w^{(t)} - \alpha \cdot \frac{\hat{m}^{(t)}}{\sqrt{\hat{v}^{(t)}} + \epsilon}
$$

## Implementation

### Data Processing

To retain only quality strokes, any strokes below a certain speed were filtered out. Additional features:
- Variance across the eight rowers for each feature
- Cube root transformation on power and work (due to cubic relationship)

### Parameters

- **Minimum Speed (m/s):** 4.75, 5.00  
- **Epochs:** 2000  
- **Weight Decay:** 1×10⁻⁷  
- **Learning Rate:** 5×10⁻⁴  
- **Batch Size:** 16  
- **Momentum:** Adam defaults (β₁ = 0.9, β₂ = 0.999)  
- **Dropout Rate:** 0.2  
- **Training/Test Split:** 80% / 20%

## Results

The goal was to predict speed within 1 second per 500m (about 1% MAPE).

### Reference Table: Speeds and Intensities

| Speed (m/s) | Pace per 500m | Intensity        |
|-------------|----------------|------------------|
| 4.50        | 1:51.1         | Low              |
| 4.75        | 1:45.0         | Steady State     |
| 5.00        | 1:40.0         | Medium Rowing    |
| 5.25        | 1:35.0         | Hard Rowing      |
| 5.50        | 1:31.0         | Race Effort      |

### Minimum Speed: 4.75 m/s

| Metric                  | Value   |
|-------------------------|---------|
| Number of data points   | 1678    |
| MSE                     | 0.0048  |
| R² Score                | 0.8261  |
| MAPE                    | 1.02%   |
| Split error (500m)      | 1.00 s  |

![MSE Loss vs. Epochs](output2.png)  
![Scatterplot for strokes ≥ 4.75 m/s](output.png)

### Minimum Speed: 5.00 m/s

| Metric                  | Value   |
|-------------------------|---------|
| Number of data points   | 1057    |
| MSE                     | 0.0017  |
| R² Score                | 0.7511  |
| MAPE                    | 0.68%   |
| Split error (500m)      | 0.65 s  |

![Scatterplot for strokes ≥ 5 m/s](output3.png)

### Extra: 5.00 m/s with Previous Stroke Speed

| Metric                  | Value   |
|-------------------------|---------|
| Number of data points   | 1055    |
| MSE                     | 0.0013  |
| R² Score                | 0.8056  |
| MAPE                    | 0.53%   |
| Split error (500m)      | 0.56 s  |

![Scatterplot with prior stroke speed](output4.png)

## Conclusion

In conclusion, we were successful in our target of predicting boat speed to under a 1-second split error over 500m. The data was filtered, and features were engineered that led to a successful neural network. Further improvements could include:
- Using grid search for hyperparameter tuning
- Collecting more quality data to reduce overfitting
