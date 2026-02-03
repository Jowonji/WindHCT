Key Differences from WiSoSuper
Input variables

WiSoSuper constructs two-channel wind fields (u, v) derived from wind speed and direction.

This implementation uses wind speed at 100 m (windspeed_100m) only, producing single-channel HR/LR pairs.

This choice simplifies the data representation and aligns with common single-channel super-resolution benchmarks.

Data storage format

WiSoSuper stores datasets as PNG images and/or TFRecords.

This implementation stores data as NumPy arrays (.npy).

This format enables direct and efficient loading in PyTorch-based super-resolution models without additional image decoding or framework-specific pipelines.

Note on normalization

The dataset is stored in physical units (m/s).
During model training, inputs are min–max normalized, and model outputs are de-normalized back to m/s for inference and evaluation.
