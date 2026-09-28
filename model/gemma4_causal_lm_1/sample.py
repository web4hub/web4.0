from absl import app
from absl import flags

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

import jax
import jax.numpy as jnp
import sentencepiece as spm


_FLAG_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint",
    None,
    required=True,
    help="Path to checkpoint.",
)

_FLAG_PATH_TOKENIZER = flags.DEFINE_string(
    "path_tokenizer",
    None,
    required=True,
    help="Path to tokenizer.",
)

_FLAG_TOTAL_SAMPLING_STEPS = flags.DEFINE_integer(
    "total_sampling_steps",
    128,
    help="Maximum number of steps to run when decoding.",
)

_FLAG_STRING_TO_SAMPLE = flags.DEFINE_string(
    "string_to_sample",
    "Where is Paris?",
    help="Input string to sample.",
)

_PADDING_MARGIN = 128
_CACHE_SIZE = 1024
_BUFFER_SIZE = 128


def _load_parameters(path_checkpoint: str) -> params_lib.Params:
    print(f"Loading checkpoint: {path_checkpoint}")

    params = params_lib.load_params(path_checkpoint)

    # Convert checkpoint arrays into JAX arrays.
    param_state = jax.tree_util.tree_map(
        jnp.array,
        params,
    )

    # Convert checkpoint parameter names/layout
    # into the format expected by the transformer.
    remapped_params = params_lib.param_remapper(
        param_state
    )

    return params_lib.nest_params(
        remapped_params
    )


def _load_and_sample(
    *,
    path_checkpoint: str,
    path_tokenizer: str,
    input_string: str,
    total_sampling_steps: int,
    cache_size: int,
    buffer_size: int,
) -> None:

    # ---------------------------------------------------------
    # 1. Load model parameters
    # ---------------------------------------------------------

    parameters = _load_parameters(
        path_checkpoint
    )

    print("Parameters loaded.")

    # ---------------------------------------------------------
    # 2. Load SentencePiece tokenizer
    # ---------------------------------------------------------

    vocab = spm.SentencePieceProcessor()

    if not vocab.Load(path_tokenizer):
        raise RuntimeError(
            f"Failed to load tokenizer: {path_tokenizer}"
        )

    print(
        f"Tokenizer loaded. "
        f"Vocabulary size: {vocab.GetPieceSize()}"
    )

    # ---------------------------------------------------------
    # 3. Build transformer configuration
    # ---------------------------------------------------------

    transformer_config = (
        transformer_lib.TransformerConfig.from_params(
            parameters,
            num_embed=(
                vocab.GetPieceSize()
                + _PADDING_MARGIN
            ),
        )
    )

    # ---------------------------------------------------------
    # 4. Create sampler
    # ---------------------------------------------------------

    sampler = sampler_lib.Sampler(
        transformer_config=transformer_config,
        vocab=vocab,
        params=parameters["transformer"],
        cache_size=cache_size,
        buffer_size=buffer_size,
        total_sampling_steps=total_sampling_steps,
    )

    # ---------------------------------------------------------
    # 5. Generate text
    # ---------------------------------------------------------

    print(f"Input string: {input_string}")

    result = sampler(
        input_strings=[input_string]
    )

    sampled_str = result.text

    print(f"Sampled string: {sampled_str}")


def main(_) -> None:
    _load_and_sample(
        path_checkpoint=_FLAG_PATH_CHECKPOINT.value,
        path_tokenizer=_FLAG_PATH_TOKENIZER.value,
        input_string=_FLAG_STRING_TO_SAMPLE.value,
        total_sampling_steps=(
            _FLAG_TOTAL_SAMPLING_STEPS.value
        ),
        cache_size=_CACHE_SIZE,
        buffer_size=_BUFFER_SIZE,
    )


if __name__ == "__main__":
    app.run(main)
