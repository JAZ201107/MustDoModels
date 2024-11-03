import torch
from model import make_model

from torch.utils.data import DataLoader, Dataset, random_split

from dataset import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os

from utils import get_device, mask

device = get_device()


def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(src, src_mask)

    # Initialize the decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_input_mask = mask(decoder_input.size(1)).to(device)
        # calculate output
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_input_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


@torch.no_grad()
def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    print_msg,
    global_step,
    writer,
    num_samples=5,
):
    model.eval()

    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    for batch in validation_ds:
        count += 1
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)

        assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"

        model_out = greedy_decode(
            model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append(model_out_text)

        # Print the source, target and model output
        print_msg("-" * console_width)
        print_msg(f"{f'SOURCE: ':>12}{source_text}")
        print_msg(f"{f'TARGET: ':>12}{target_text}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

        if count == num_samples:
            print_msg("-" * console_width)
            break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = config.DATA.TOKENIZER_FILE.format(lang)

    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        f"{config.TRAINING}",
        f"{config.DATA.LANG_SRC}-{config.DATA.LANG_TGT}",
        split="train",
    )

    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.DATA.LANG_SRC)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.DATA.LANG_TGT)

    # Keep 90% of the data for training
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(
    config,
    vocab_src_len,
    vocab_tgt_len,
):
    model = make_model(
        vocab_src_len,
        vocab_tgt_len,
        d_model=config["d_model"],
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )

    return model


def train_model(config):
    pass


if __name__ == "__main__":
    pass
