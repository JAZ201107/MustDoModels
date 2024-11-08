import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


from utils import causal_mask


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]

        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = (
            self.seq_len - len(dec_input_tokens) - 1
        )  # Only add <SOS> token to the decoder input

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(
                f"The Sentence is too long for the model. Max length is {self.seq_len - 2} tokens."
            )

        # Add <SOS> <EOS> <PAD> tokens to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Add <SOS> <PAD> tokens to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def get_all_senteces(ds, lang):
    for example in ds:
        yield example["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config.DATA.TOKENIZER_FILE.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=[f"[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_senteces(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataloader(config):
    ds_raw = load_dataset(
        f"{config.DATA.DATASOURCE}",
        f"{config.DATA.LANG_SRC}-{config.DATA.LANG_TGT}",
        split="train",
    )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.DATA.LANG_SRC)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.DATA.LANG_TGT)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Build Dataset
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.DATA.LANG_SRC,
        config.DATA.LANG_TGT,
        config.MODEL.SEQ_LEN,
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.DATA.LANG_SRC,
        config.DATA.LANG_TGT,
        config.MODEL.SEQ_LEN,
    )

    # Find the maximum length of each sentence in the source and target
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config.DATA.LANG_SRC]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config.DATA.LANG_TGT]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config.TRAINING.BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer_src": tokenizer_src,
        "tokenizer_tgt": tokenizer_tgt,
    }
