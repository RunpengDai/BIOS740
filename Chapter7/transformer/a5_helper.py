import json
import math
import time

import matplotlib.pyplot as plt
import seaborn
import torch



def get_toy_data(path: str = "final_data.json"):
    return json.load(open(path))


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_func,
    num_epochs,
    batch_size=32,
    warmup_lr=6e-6,
    warmup_interval=1000,
    lr=6e-4,
    device=torch.device("cpu"),
):
    print("Training started...")
    if warmup_interval is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=warmup_lr, betas=(0.9, 0.995), eps=1e-9
        )
    iteration = 0
    for epoch_num in range(num_epochs):
        epoch_loss = []
        model.train()
        for it in train_dataloader:
            inp, inp_pos, out, out_pos = it
            model = model.to(device)
            inp_pos = inp_pos.to(device)
            out_pos = out_pos.to(device)
            out = out.to(device)
            inp = inp.to(device)
            gnd = out[:, 1:].contiguous().view(-1).long()
            optimizer.zero_grad()

            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)
            epoch_loss.append(loss.item())
            if warmup_interval is not None and iteration == warmup_interval:
                print(
                    f"End of warmup. Swapping learning rates from {warmup_lr} to {lr}"
                )
                for param_group in optimizer.param_groups:
                    warmup_lr = lr
                    param_group["lr"] = lr

            loss.backward()
            optimizer.step()
            iteration = iteration + 1
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss, val_acc = val(model, val_dataloader, loss_func, batch_size)
        loss_hist = avg_epoch_loss / (batch_size * 4)
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]",
            "val_loss: [val_loss ",
            f"{val_loss:.4f}",
            "]",
        )

    return model


def val(model, dataloader, loss_func, batch_size, device=torch.device("cpu")):
    model.eval()
    epoch_loss = []
    num_correct = 0
    total = 0
    for it in dataloader:
        inp, inp_pos, out, out_pos = it

        model = model.to(device)
        inp_pos = inp_pos.to(device)
        out_pos = out_pos.to(device)
        out = out.to(device)
        inp = inp.to(device)
        gnd = out[:, 1:].contiguous().view(-1).long()
        pred = model(inp.long(), inp_pos, out.long(), out_pos)
        loss = loss_func(pred, gnd)

        pred_max = pred.max(1)[1]
        gnd = gnd.contiguous().view(-1)

        n_correct = pred_max.eq(gnd)
        n_correct = n_correct.sum().item()
        num_correct = num_correct + n_correct

        total = total + len(pred_max)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_epoch_loss / (batch_size * 4), n_correct / total


def inference(model, inp_exp, inp_exp_pos, out_pos_exp, out_seq_len, device):
    model.eval()
    if device =="cuda":
        y_init = torch.LongTensor([14]).unsqueeze(0).cuda().view(1, 1)
    else:
        y_init = torch.LongTensor([14]).unsqueeze(0).view(1, 1)

    ques_emb = model.emb_layer(inp_exp)
    q_emb_inp = ques_emb + inp_exp_pos
    enc_out = model.encoder(q_emb_inp)
    for i in range(out_seq_len - 1):
        ans_emb = model.emb_layer(y_init)
        a_emb_inp = ans_emb + out_pos_exp[:, : y_init.shape[1], :]
        dec_out = model.decoder(a_emb_inp, enc_out, None)
        _, next_word = torch.max(
            dec_out[0, y_init.shape[1] - 1 : y_init.shape[1]], dim=1
        )

        y_init = torch.cat([y_init, next_word.view(1, 1)], dim=1)
    return y_init, model


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x,
        square=True,
        yticklabels=y,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
    )
