import torch

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
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
