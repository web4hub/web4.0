pip install --upgrade poetry

# For inference on NVIDIA GPUs:
poetry install --with cuda

# For CPU inference:
poetry install

# For TPU inference:
poetry install --with tpu
