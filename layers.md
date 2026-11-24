# Model Layers

The model is a single `nn.Sequential` CNN defined in `drone_classifier.py`. Tensor shapes are given as **(channels × range × Doppler)** for one sample. The batch dimension is left out.

**Input:** `1 × 5 × 150`. This is one standardized Doppler map in dB: 5 range cells by 150 Doppler bins.

## Layer Table

| # | Layer | Config | Output shape | Params |
|---:|---|---|---|---:|
| 1 | Conv2d | 1→16, kernel 3×5, padding 1×2 | 16 × 5 × 150 | 256 |
| 2 | BatchNorm2d | 16 | 16 × 5 × 150 | 32 |
| 3 | ReLU | – | 16 × 5 × 150 | 0 |
| 4 | MaxPool2d | kernel 1×2 | 16 × 5 × 75 | 0 |
| 5 | Conv2d | 16→32, kernel 3×5, padding 1×2 | 32 × 5 × 75 | 7,712 |
| 6 | BatchNorm2d | 32 | 32 × 5 × 75 | 64 |
| 7 | ReLU | – | 32 × 5 × 75 | 0 |
| 8 | MaxPool2d | kernel 1×2 | 32 × 5 × 37 | 0 |
| 9 | Conv2d | 32→64, kernel 3×5, padding 1×2 | 64 × 5 × 37 | 30,784 |
| 10 | BatchNorm2d | 64 | 64 × 5 × 37 | 128 |
| 11 | ReLU | – | 64 × 5 × 37 | 0 |
| 12 | AdaptiveAvgPool2d | output 1×1 | 64 × 1 × 1 | 0 |
| 13 | Flatten | – | 64 | 0 |
| 14 | Linear | 64→2 | 2 | 130 |
| | **Total** | | | **39,106** |

## Layer Descriptions

### Conv2d (layers 1, 5, 9)
2D convolution that learns local patterns in the range-Doppler map.
- **Kernel 3×5:** spans 3 range cells and 5 Doppler bins. The kernel is wider on the Doppler axis because that is where the class information lives (the spread of micro-Doppler energy from rotors).
- **Padding 1×2:** "same" padding, so the height and width of the map stay unchanged.
- **Filters 16 → 32 → 64:** the number of filters doubles in each block, so deeper layers can represent more and more abstract spectral patterns.
- Parameters per layer = `in × out × 3 × 5 + out` (weights + bias).

### BatchNorm2d (layers 2, 6, 10)
Normalizes each channel across the batch to zero mean and unit variance, then applies a learned scale (γ) and shift (β), which gives 2 parameters per channel. This stabilizes training and allows the relatively high learning rate of 1e-3. At eval time (`model.eval()`) it uses running statistics instead of batch statistics.

### ReLU (layers 3, 7, 11)
`max(0, x)` non-linearity applied after each normalized convolution. It has no parameters.

### MaxPool2d (1×2) (layers 4, 8)
Takes the maximum over pairs of adjacent Doppler bins and **halves only the Doppler axis** (150 → 75 → 37). The range axis is not pooled because it has only 5 cells. Pooling:
- reduces computation for later layers,
- gives some tolerance to small Doppler shifts,
- widens the receptive field along Doppler.

There is no pool after the third block, because global pooling comes next.

### AdaptiveAvgPool2d(1) (layer 12)
Global average pooling. It averages each of the 64 feature maps (5 × 37) down to a single value. The model therefore does not care *where* a pattern appears in the map, only *how strongly* it is present. It also keeps the classifier tiny: the alternative, flattening 64×5×37 = 11,840 values into the linear layer, would need many more parameters.

### Flatten (layer 13)
Reshapes `64 × 1 × 1` to a 64-element vector for the linear layer.

### Linear (layer 14)
A fully connected layer that maps the 64 pooled features to **2 logits** (not drone, drone). There is no softmax in the model: `CrossEntropyLoss` applies log-softmax during training, and prediction uses `argmax` over the logits.

## Receptive Field

The receptive field is the region of the input that one unit after the third conv block can see.

| Axis | Receptive field | Input size |
|---|---:|---:|
| Range | 7 cells | 5 (covers all of it) |
| Doppler | 32 bins | 150 |

Each final feature therefore sees all range cells and about a fifth of the Doppler spectrum. Global average pooling then combines these local Doppler patterns across the whole spectrum.

## Parameter Breakdown

| Component | Params | Share |
|---|---:|---:|
| Conv layers | 38,752 | 99.1% |
| BatchNorm layers | 224 | 0.6% |
| Linear layer | 130 | 0.3% |

Almost all of the capacity is in the convolution layers. The third conv alone holds about 79% of the parameters.
