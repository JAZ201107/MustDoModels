import torch

from utils import get_device, causal_mask


def greedy_decode(
    model,
    source,
    source_mask,
    tokenizer_tgt,
    max_len=100,
):
    device = get_device()

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Pre-compute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:-1])
        _, next_word = prob.max(dim=-1)
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


def beam_search_decoder(
    model,
    beam_size,
    source,
    source_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len=100,
):
    device = get_device()

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type(source).to(device)

    candidates = [(decoder_initial_input, 1)]

    while True:
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the <EOS> token
            if candidate[0, -1] == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = (
                causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            )

            # Calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)

            # Get next token probabilities
            prob = model.project(out[:, -1])
            # Get the top k tokens
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)

            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_pron = topk_prob[0][i].item()

                # Create new candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                new_candidates.append((new_candidate, score + token_pron))

        # Sort the new candidates by score
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Get the top k candidates
        candidates = new_candidates[:beam_size]

        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    return candidates[0][0].squeeze(0)
