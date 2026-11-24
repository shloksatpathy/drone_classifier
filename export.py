"""E9: fold FiLM for one radar into BatchNorm, post-training INT8 quantization (FX, x86), compare with
FP32 on the session-split test set, export the FP32 folded model to ONNX.
Usage: python export.py models/saab.pt [radar]"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from config import wvec
from data import load_saab, session_split
from model import Net
from train import real_pool, inputs

path = sys.argv[1]
radar = sys.argv[2] if len(sys.argv) > 2 else "saab"
ck = torch.load(path, weights_only=False)
net = Net(ck["size"])
net.load_state_dict(ck["state"])
net.eval()


class Folded(nn.Module):
    """Plain conv-BN-ReLU network for a fixed waveform vector. Output: [z, h1, h2, h3]."""

    def __init__(self, net, w):
        super().__init__()
        layers = []
        for i in range(3):
            bn = nn.BatchNorm1d(net.bns[i].num_features)
            bn.load_state_dict(net.bns[i].state_dict())
            g, b = net.films[i](w[None])[0].chunk(2)
            with torch.no_grad():
                bn.weight.mul_(1 + g)
                bn.bias.mul_(1 + g).add_(b)
            layers += [net.convs[i], bn, nn.ReLU()] + ([nn.MaxPool1d(2)] if i < 2 else [])
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.heads = nn.Linear(net.h1.in_features, 19)
        with torch.no_grad():
            self.heads.weight.copy_(torch.cat([net.h1.weight, net.h2.weight, net.h3.weight]))
            self.heads.bias.copy_(torch.cat([net.h1.bias, net.h2.bias, net.h3.bias]))

    def forward(self, x):
        z = self.body(x)
        return torch.cat([z, self.heads(z)], 1)


with torch.no_grad():
    fp32 = Folded(net, torch.tensor(wvec(radar))).eval()

d = load_saab()
sess = session_split(d["name"], d["session"])
cpu_pool = lambda idx: {k: v.cpu() for k, v in real_pool(d, idx).items()}
net.cpu()


def xs(idx):
    p = cpu_pool(idx)
    return inputs(net, p, torch.arange(len(idx)), None, False)[0].unsqueeze(1)


x_cal = xs(np.random.default_rng(0).choice(np.where(sess == 1)[0], 2000, replace=False))
te = np.where(sess == 3)[0]
x_te, y_te = xs(te), d["cls"][te]

torch.backends.quantized.engine = "x86"
q = prepare_fx(Folded(net, torch.tensor(wvec(radar))).eval(), get_default_qconfig_mapping("x86"), (x_cal[:1],))
with torch.no_grad():
    q(x_cal)
    q = convert_fx(q)
    c = net.h1.in_features
    o32, o8 = fp32(x_te), q(x_te)
    with torch.no_grad():
        ref = net(x_te.squeeze(1), torch.tensor(wvec(radar)).expand(len(te), 4))["h1"]
a_ref, a32, a8 = [(o.argmax(1).numpy() == y_te).mean() for o in (ref, o32[:, c:c + 3], o8[:, c:c + 3])]
print(f"E9 {radar}: acc3 original {a_ref:.4f}, folded FP32 {a32:.4f}, INT8 {a8:.4f}, drop {a32 - a8:+.4f}")
print(f"   H1 agreement INT8 vs FP32: {(o32[:, c:c + 3].argmax(1) == o8[:, c:c + 3].argmax(1)).float().mean():.4f}")

out = path.replace(".pt", f"_{radar}.onnx")
torch.onnx.export(fp32, (x_cal[:1],), out, input_names=["spectrum"], output_names=["z_h1_h2_h3"],
                  dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
torch.jit.save(torch.jit.trace(q, x_cal[:1]), path.replace(".pt", f"_{radar}_int8.pt"))
print("wrote", out, "and", path.replace(".pt", f"_{radar}_int8.pt"))
