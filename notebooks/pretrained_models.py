from __future__ import absolute_import, division, print_function

import sys
import os
import collections

import torch


def get_feature_model(args):
    activations = collections.OrderedDict()

    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o

        return hook_fn
    if args['features'] == "cpc":
        sys.path.append("../dpc")
        sys.path.append("../backbone")
        from model_3d import DPC_RNN

        model = DPC_RNN(
                        sample_size=64, num_seq=8, seq_len=5, network="monkeynet", pred_step=3
                        )
        checkpoint = torch.load(args['PATH'])
        subnet_dict = extract_subnet_dict(checkpoint["state_dict"])

        model.load_state_dict(subnet_dict)
        model = model.backbone
        layers = collections.OrderedDict(
            [(f"layer{i:02}", l[-1]) for i, l in enumerate(model.layers)]
        )
        metadata = {"sz": 64, "threed": True}
    else:
        raise NotImplementedError("Model not implemented yet")

    for key, layer in layers.items():
        layer.register_forward_hook(hook(key))

    metadata["layers"] = layers

        # Put model in eval mode (for batch_norm, dropout, etc.)
    model.eval()
    return model, activations, metadata


def extract_subnet_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith("subnet.") or k.startswith("module."):
            out[k[7:]] = v
    return out


def get_model_layers(model, getLayerRepr=False):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers

if __name__ == '__main__':
    

    PATH = '/network/tmp1/bakhtias/Results/log_monkeynet_ucf101_path_1_1/ucf101-64_rnet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch100.pth.tar'
    args = {'arch': 'VisalNet',
             'PATH': PATH,
             'slowfast_root': '../../slowfast',
             'ntau': 1,
             'subsample_layers': False}
    
    model = get_feature_model(args)
    
    