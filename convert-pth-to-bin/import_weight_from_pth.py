import json
from copy import deepcopy
from torch import nn
from focalnet import FocalModulation, FocalNetBlock, BasicLayer, PatchEmbed, FocalNet


def map_layernorm(dictionary: dict, model: nn.LayerNorm) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["gamma"]["param"]["value"] = model.weight.data.flatten().tolist()
    dictionary["beta"]["param"]["value"] = model.bias.data.flatten().tolist()

    return dictionary


def map_linear(dictionary: dict, model: nn.Linear) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["weight"]["param"]["value"] = model.weight.data.T.flatten().tolist()
    if dictionary["bias"] is not None and model.bias is not None:
        dictionary["bias"]["param"]["value"] = model.bias.data.flatten().tolist()

    return dictionary


def map_conv(dictionary: dict, model: nn.Conv2d) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["weight"]["param"]["value"] = model.weight.flatten().tolist()
    if dictionary["bias"] is not None and model.bias is not None:
        dictionary["bias"]["param"]["value"] = model.bias.flatten().tolist()

    return dictionary


def map_patch_embed(dictionary: dict, model: PatchEmbed) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["proj"] = map_conv(dictionary["proj"], model.proj)
    dictionary["norm"] = map_layernorm(dictionary["norm"], model.norm)

    return dictionary


def map_modulation(dictionary: dict, model: FocalModulation) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["pre_linear"] = map_linear(dictionary["pre_linear"], model.f)
    dictionary["mix_channel"] = map_conv(dictionary["mix_channel"], model.h)
    dictionary["post_linear"] = map_linear(dictionary["post_linear"], model.proj)
    dictionary["focal_layers"][0]["layers"][0]["Conv2d"] = map_conv(
        dictionary["focal_layers"][0]["layers"][0]["Conv2d"], model.focal_layers[0][0]
    )
    dictionary["focal_layers"][1]["layers"][0]["Conv2d"] = map_conv(
        dictionary["focal_layers"][1]["layers"][0]["Conv2d"], model.focal_layers[1][0]
    )

    return dictionary


def map_block(dictionary: dict, model: FocalNetBlock) -> dict:
    dictionary = deepcopy(dictionary)
    dictionary["modulation"] = map_modulation(
        dictionary["modulation"], model.modulation
    )
    dictionary["mlp"]["linear1"] = map_linear(
        dictionary["mlp"]["linear1"], model.mlp.fc1
    )
    dictionary["mlp"]["linear2"] = map_linear(
        dictionary["mlp"]["linear2"], model.mlp.fc2
    )
    dictionary["layernorm1"] = map_layernorm(dictionary["layernorm1"], model.norm1)
    dictionary["layernorm2"] = map_layernorm(dictionary["layernorm2"], model.norm2)

    return dictionary


def map_layer(dictionary: dict, model: BasicLayer) -> dict:
    dictionary = deepcopy(dictionary)
    depth = len(model.blocks)
    for i in range(depth):
        dictionary["blocks"][i] = map_block(dictionary["blocks"][i], model.blocks[i])

    if dictionary["downsample"] is not None and model.downsample is not None:
        dictionary["downsample"] = map_patch_embed(
            dictionary["downsample"], model.downsample
        )

    return dictionary


def map_focal_net(dictionary: dict, model: FocalNet):
    dictionary = deepcopy(dictionary)
    dictionary["patch_embed"] = map_patch_embed(
        dictionary["patch_embed"], model.patch_embed
    )
    num_layers = len(model.layers)
    for i in range(num_layers):
        dictionary["layers"][i] = map_layer(dictionary["layers"][i], model.layers[i])
    dictionary["layernorm"] = map_layernorm(dictionary["layernorm"], model.norm)
    if dictionary["head"] is not None and model.head is not None:
        dictionary["head"] = map_linear(dictionary["head"], model.head)

    return dictionary


if __name__ == "__main__":
    from focalnet import focalnet_small_lrf

    model = focalnet_small_lrf(
        pretrained=True,
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
    )
    model.eval()

    with open("./template.json") as f:
        focal_net = json.load(f)

    focal_net["item"] = map_focal_net(focal_net["item"], model)

    with open("./template.json", "w") as f:
        json.dump(focal_net, f)
