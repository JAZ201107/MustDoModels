import torch
from utils import save_checkpoint, load_checkpoint
from tqdm import tqdm


@torch.no_grad()
def validation_one_epoch(model, data_loader, criterion, config):
    model.eval()


def train_one_epoch(model, train_dataloader, criterion, optimizer, config, epoch):
    torch.cuda.empty_cache()
    model.train()

    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:

        encoder_input = batch["encoder_input"].to(config.device)
        decoder_input = batch["decoder_input"].to(config.device)
        encoder_mask = batch["encoder_mask"].to(config.device)
        decoder_mask = batch["decoder_mask"].to(config.device)

        # Run the tensors through the encoder, decoder and the projection layer
        decoder_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
        proj_out = model.generator(decoder_output)

        label = batch["label"].to(config.device)

        loss = criterion(
            proj_out.view(
                -1,
            ),
            label.view(-1),
        )


def train(config):
    pass
