from dataclasses import dataclass

from config import Config, map_options
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer, IntegerTokenizer, Tokenizer


@dataclass
class GPTConfig(Config):
    block_size: int = 0  # max sequence length
    vocab_size: int = 0  # number of tokens
    n_layer: int = 0  # number of layers
    n_head: int = 0  # number of heads
    n_embd: int = 0  # embedding dimension

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Infer tokenizer from vocabulary size.
        """
        match self.vocab_size:
            case TikTokenTokenizer.vocab_size:
                return TikTokenTokenizer()
            case ASCIITokenizer.vocab_size:
                return ASCIITokenizer()
            case _:
                # For any other vocab size, assume it's an integer tokenizer
                return IntegerTokenizer(self.vocab_size)

    @staticmethod
    def dict_factory(fields: list) -> dict:
        """
        Only export integer fields (exclude name and device)
        """
        return {k: v for (k, v) in fields if type(v) is int}


# GPT configuration options
gpt_options: dict[str, GPTConfig] = map_options(
    GPTConfig(
        name="ascii_64x4",
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=64,
    ),
    GPTConfig(
        name="ascii_128x6",
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=6,
        n_head=4,
        n_embd=128,
    ),
    GPTConfig(
        name="tiktoken_32x4",
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=32,
    ),
    GPTConfig(
        name="tiktoken_256x4",
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=256,
    ),
    GPTConfig(
        name="integer_1000_128x4",
        block_size=128,
        vocab_size=1000,
        n_layer=4,
        n_head=8,
        n_embd=128,
    ),
    GPTConfig(
        name="integer_10000_256x6",
        block_size=128,
        vocab_size=10000,
        n_layer=6,
        n_head=16,
        n_embd=256,
    ),
    GPTConfig(
        name="integer_100_64x1",
        block_size=32,
        vocab_size=100,
        n_layer=1,
        n_head=4,
        n_embd=64,
    ),
    GPTConfig(
        name="mess3_64x1",
        block_size=10,
        vocab_size=4,
        n_layer=1,
        n_head=1,
        n_embd=64,
    ),
    GPTConfig(
        name="mess3_12_64x1",
        block_size=12,
        vocab_size=4,
        n_layer=1,
        n_head=1,
        n_embd=64,
    ),
)
