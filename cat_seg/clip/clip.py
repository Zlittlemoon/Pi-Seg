import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

# from .model import build_model
from .model_vpt import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    return list(_MODELS.keys())


def _build_clip_transform(n_px: int):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def _unwrap_checkpoint(ckpt):
    """
    Support common checkpoint wrappers:
      {"state_dict": ...}
      {"model": ...}
      {"module": ...}
      {"model_state_dict": ...}
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ["state_dict", "model", "module", "model_state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    return ckpt


def _clean_checkpoint_keys(state_dict):
    """
    Remove common prefixes and normalize several OpenCLIP-style names.
    """
    new_sd = {}

    for k, v in state_dict.items():
        nk = k

        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        # common OpenCLIP prefix
        if nk.startswith("visual.trunk."):
            nk = "visual." + nk[len("visual.trunk."):]

        new_sd[nk] = v

    return new_sd


def _maybe_add_visual_prefix(state_dict, model):
    """
    Some vision-only checkpoints may store keys like:
        conv1.weight
        positional_embedding
        transformer.resblocks...

    CAT-Seg CLIP expects:
        visual.conv1.weight
        visual.positional_embedding
        visual.transformer.resblocks...

    This function adds "visual." only when doing so increases matching keys.
    """
    model_keys = set(model.state_dict().keys())

    matched = sum(1 for k in state_dict.keys() if k in model_keys)
    if matched > 0:
        return state_dict

    visual_prefixed = {}
    for k, v in state_dict.items():
        if (
            k.startswith("visual.")
            or k.startswith("transformer.")
            or k.startswith("token_embedding.")
            or k.startswith("ln_final.")
            or k.startswith("text_projection")
            or k.startswith("positional_embedding")
            or k.startswith("logit_scale")
        ):
            visual_prefixed[k] = v
        else:
            visual_prefixed["visual." + k] = v

    matched_visual = sum(1 for k in visual_prefixed.keys() if k in model_keys)

    if matched_visual > matched:
        print(
            f"[LAST-ViT] auto add visual. prefix: matched {matched} -> {matched_visual}",
            flush=True,
        )
        return visual_prefixed

    return state_dict


def _convert_qkv_between_clip_formats(state_dict, model):
    """
    Convert attention QKV weights between common CLIP formats.

    Format A, packed:
        xxx.attn.in_proj_weight
        xxx.attn.in_proj_bias

    Format B, separated:
        xxx.attn.q_proj_weight
        xxx.attn.k_proj_weight
        xxx.attn.v_proj_weight
        xxx.attn.q_proj_bias
        xxx.attn.k_proj_bias
        xxx.attn.v_proj_bias

    Your current log shows CAT-Seg's model expects separated q/k/v weights,
    while LAST-ViT/OpenCLIP checkpoint may provide packed in_proj_weight.
    This function supports both directions.
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    out = dict(state_dict)

    converted_packed_to_split = 0
    converted_split_to_packed = 0

    # ------------------------------------------------------------
    # 1) packed in_proj_weight / in_proj_bias -> separated q/k/v
    # ------------------------------------------------------------
    for k, v in list(state_dict.items()):
        if k.endswith(".attn.in_proj_weight"):
            prefix = k[:-len("in_proj_weight")]

            q_key = prefix + "q_proj_weight"
            k_key = prefix + "k_proj_weight"
            v_key = prefix + "v_proj_weight"

            if (
                q_key in model_keys
                and k_key in model_keys
                and v_key in model_keys
                and q_key not in out
                and k_key not in out
                and v_key not in out
            ):
                if v.shape[0] % 3 == 0:
                    q_w, k_w, v_w = v.chunk(3, dim=0)
                    out[q_key] = q_w
                    out[k_key] = k_w
                    out[v_key] = v_w
                    converted_packed_to_split += 3

        if k.endswith(".attn.in_proj_bias"):
            prefix = k[:-len("in_proj_bias")]

            q_key = prefix + "q_proj_bias"
            k_key = prefix + "k_proj_bias"
            v_key = prefix + "v_proj_bias"

            if (
                q_key in model_keys
                and k_key in model_keys
                and v_key in model_keys
                and q_key not in out
                and k_key not in out
                and v_key not in out
            ):
                if v.shape[0] % 3 == 0:
                    q_b, k_b, v_b = v.chunk(3, dim=0)
                    out[q_key] = q_b
                    out[k_key] = k_b
                    out[v_key] = v_b
                    converted_packed_to_split += 3

    # ------------------------------------------------------------
    # 2) separated q/k/v -> packed in_proj_weight / in_proj_bias
    # ------------------------------------------------------------
    prefixes = set()
    for k in state_dict.keys():
        if k.endswith(".attn.q_proj_weight"):
            prefixes.add(k[:-len("q_proj_weight")])

    for prefix in prefixes:
        q_key = prefix + "q_proj_weight"
        k_key = prefix + "k_proj_weight"
        v_key = prefix + "v_proj_weight"
        packed_key = prefix + "in_proj_weight"

        if (
            packed_key in model_keys
            and packed_key not in out
            and q_key in state_dict
            and k_key in state_dict
            and v_key in state_dict
        ):
            out[packed_key] = torch.cat(
                [state_dict[q_key], state_dict[k_key], state_dict[v_key]],
                dim=0,
            )
            converted_split_to_packed += 1

        q_bias_key = prefix + "q_proj_bias"
        k_bias_key = prefix + "k_proj_bias"
        v_bias_key = prefix + "v_proj_bias"
        packed_bias_key = prefix + "in_proj_bias"

        if (
            packed_bias_key in model_keys
            and packed_bias_key not in out
            and q_bias_key in state_dict
            and k_bias_key in state_dict
            and v_bias_key in state_dict
        ):
            out[packed_bias_key] = torch.cat(
                [state_dict[q_bias_key], state_dict[k_bias_key], state_dict[v_bias_key]],
                dim=0,
            )
            converted_split_to_packed += 1

    print(
        "[LAST-ViT] QKV conversion: "
        f"packed->split tensors added={converted_packed_to_split}, "
        f"split->packed tensors added={converted_split_to_packed}",
        flush=True,
    )

    return out


def _filter_loadable_tensors(state_dict, model):
    """
    Keep only tensors whose keys exist in the target model and whose shapes match.
    This avoids crashing on classifier heads or architecture-specific parameters.
    """
    model_sd = model.state_dict()

    loadable = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue

        if tuple(v.shape) == tuple(model_sd[k].shape):
            loadable[k] = v
        else:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    return loadable, skipped_missing, skipped_shape


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
    prompt_depth=0,
    prompt_length=0,
):
    name = os.path.expanduser(name)

    # ------------------------------------------------------------------
    # Case 1: local LAST-ViT / local CLIP checkpoint path
    # ------------------------------------------------------------------
    if os.path.isfile(name):
        local_ckpt_path = os.path.abspath(name)
        print(f"[CLIP] Local checkpoint detected: {local_ckpt_path}", flush=True)

        # Build a normal OpenAI CLIP ViT-B/16 first.
        # This preserves the text encoder required by CAT-Seg.
        base_name = "ViT-B/16"
        base_model_path = _download(_MODELS[base_name])
        base_jit_model = torch.jit.load(
            base_model_path,
            map_location=device if jit else "cpu",
        ).eval()

        n_px = base_jit_model.input_resolution.item()
        transform = _build_clip_transform(n_px)

        model = build_model(
            base_jit_model.state_dict(),
            prompt_depth,
            prompt_length,
        ).to(device)

        ckpt = torch.load(local_ckpt_path, map_location="cpu")

        state_dict = _unwrap_checkpoint(ckpt)
        original_ckpt_tensor_count = len(state_dict)

        state_dict = _clean_checkpoint_keys(state_dict)
        state_dict = _maybe_add_visual_prefix(state_dict, model)
        state_dict = _convert_qkv_between_clip_formats(state_dict, model)

        converted_tensor_count = len(state_dict)

        loadable, skipped_missing, skipped_shape = _filter_loadable_tensors(
            state_dict,
            model,
        )

        missing, unexpected = model.load_state_dict(loadable, strict=False)

        print(f"[LAST-ViT] original ckpt tensors: {original_ckpt_tensor_count}", flush=True)
        print(f"[LAST-ViT] converted ckpt tensors: {converted_tensor_count}", flush=True)
        print(f"[LAST-ViT] loaded tensors: {len(loadable)} / converted ckpt tensors: {converted_tensor_count}", flush=True)
        print(f"[LAST-ViT] missing keys after load: {len(missing)}", flush=True)
        print(f"[LAST-ViT] unexpected keys after load: {len(unexpected)}", flush=True)
        print(f"[LAST-ViT] skipped missing-in-model keys: {len(skipped_missing)}", flush=True)
        print(f"[LAST-ViT] skipped shape-mismatch keys: {len(skipped_shape)}", flush=True)
        print("[LAST-ViT] first 20 missing after load:", missing[:20], flush=True)
        print("[LAST-ViT] first 20 skipped missing-in-model:", skipped_missing[:20], flush=True)
        print("[LAST-ViT] first 20 skipped shape-mismatch:", skipped_shape[:20], flush=True)

        # This is a useful sanity check for your current issue.
        qkv_missing = [
            k for k in missing
            if (
                ".attn.q_proj_weight" in k
                or ".attn.k_proj_weight" in k
                or ".attn.v_proj_weight" in k
                or ".attn.q_proj_bias" in k
                or ".attn.k_proj_bias" in k
                or ".attn.v_proj_bias" in k
            )
        ]
        print(f"[LAST-ViT] q/k/v missing keys after conversion: {len(qkv_missing)}", flush=True)
        print("[LAST-ViT] first 20 q/k/v missing:", qkv_missing[:20], flush=True)

        return model, transform

    # ------------------------------------------------------------------
    # Case 2: user gave a path-like string, but the file does not exist
    # ------------------------------------------------------------------
    if name.endswith((".pt", ".pth", ".bin", ".ckpt")) or "/" in name:
        raise RuntimeError(
            f"Local checkpoint path does not exist: {name}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please use an absolute path or put the file under this directory."
        )

    # ------------------------------------------------------------------
    # Case 3: normal OpenAI CLIP model name
    # ------------------------------------------------------------------
    if name not in _MODELS:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    model_path = _download(_MODELS[name])
    model = torch.jit.load(
        model_path,
        map_location=device if jit else "cpu",
    ).eval()

    n_px = model.input_resolution.item()
    transform = _build_clip_transform(n_px)

    if not jit:
        model = build_model(
            model.state_dict(),
            prompt_depth,
            prompt_length,
        ).to(device)
        return model, transform

    return model, transform


def load_custom(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
    n_px=224,
):
    if name not in _MODELS:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    model_path = _download(_MODELS[name])
    model = torch.jit.load(
        model_path,
        map_location=device if jit else "cpu",
    ).eval()

    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    if not jit:
        model = build_model(model.state_dict()).to(device)
        return model, transform

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)),
        example_inputs=[],
    )
    device_node = [
        n for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if (
                    "value" in node.attributeNames()
                    and str(node["value"]).startswith("cuda")
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if device == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(),
            example_inputs=[],
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, transform


def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [
        [sot_token] + _tokenizer.encode(text) + [eot_token]
        for text in texts
    ]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(
                f"Input {texts[i]} is too long for context length {context_length}"
            )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result