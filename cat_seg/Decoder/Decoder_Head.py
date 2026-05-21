# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import fvcore.nn.weight_init as weight_init
import open_clip
import os
import re

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .Cost_Aggregation import Aggregator
from cat_seg.clip import clip
from cat_seg.clip import imagenet_templates
from cat_seg.pini_vpn import ImageVpnGenerator, TextVpnGenerator
from cat_seg.visualizer import CATSegVisualizer
from cat_seg.lazy_score import compute_lazy_score


class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        pini_enabled: bool = True,
        pini_image_vpn_enabled: bool = True,
        pini_text_vpn_enabled: bool = True,
        pini_reduction: int = 1,
        pini_text_noise_std: float = 0.02,

        # ===== image noise =====
        pini_image_noise_type: str = "laplace",
        pini_image_student_t_df: float = 3.0,

        # ===== text noise =====
        pini_text_noise_type: str = "uniform",
        pini_text_student_t_df: float = 3.0,

        is_vis: bool = False,
        vis_save_dir: str = "./delta_corr_vis",
    ):
        """
        Args:
            
        """
        super().__init__()
        
        self.is_vis = is_vis
        self.visualizer = CATSegVisualizer(vis_save_dir)
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer

        # PiNI: feature-level noise generators
        self.image_vpn = None
        self.text_vpn = None
        if pini_enabled:
            if pini_image_vpn_enabled:
                self.image_vpn = ImageVpnGenerator(
                    clip_dim=text_guidance_dim,
                    reduction=pini_reduction,
                    noise_type=pini_image_noise_type,
                    student_t_df=pini_image_student_t_df,
                )
            if pini_text_vpn_enabled:
                self.text_vpn = TextVpnGenerator(
                    clip_dim=text_guidance_dim,
                    noise_std=pini_text_noise_std,
                    noise_type=pini_text_noise_type,
                    student_t_df=pini_text_student_t_df,
                )

        self.tokens = None
        self.cache = None
        self.cached_corr_before = None
        self.cached_corr_after = None
        self.cached_delta_corr = None

        # ============================================================
        # SegEarth-OV style Global Bias Alleviation
        # ============================================================
        self.segearth_gba = os.environ.get("PISEG_SEGEARTH_GBA", "0") == "1"
        self.segearth_gba_lambda = float(os.environ.get("PISEG_SEGEARTH_LAMBDA", "-0.3"))

        # 默认只在 eval / inference 使用，符合 SegEarth-OV training-free 思路。
        # 如果要训练时也启用，设置 PISEG_SEGEARTH_APPLY_TRAIN=1
        self.segearth_apply_train = os.environ.get("PISEG_SEGEARTH_APPLY_TRAIN", "0") == "1"

        self.segearth_verbose = os.environ.get("PISEG_SEGEARTH_VERBOSE", "0") == "1"

        print(
            "[SegEarth-GBA-Config] "
            f"enabled={self.segearth_gba}, "
            f"lambda={self.segearth_gba_lambda}, "
            f"apply_train={self.segearth_apply_train}, "
            f"verbose={self.segearth_verbose}",
            flush=True,
        )

        # ============================================================
        # Uncertainty-aware Lazy Image-SPM Gate
        # ============================================================
        # 默认关闭；只有显式设置 PISEG_UA_LAZY_GATE=1 才启用。
        self.ua_lazy_gate = os.environ.get("PISEG_UA_LAZY_GATE", "0") == "1"

        # 四种 gate 模式：
        #   none     : 不使用 gate，即使 PISEG_UA_LAZY_GATE=1 也等价 baseline
        #   unc      : 只使用 Raw cost uncertainty
        #   lazy     : 只使用 LazyScore
        #   unc_lazy : 使用 uncertainty * LazyScore
        self.ua_lazy_gate_type = os.environ.get(
            "PISEG_UA_LAZY_GATE_TYPE",
            "unc_lazy",
        ).lower()

        # gate 强度。gate_map = 1 + alpha * gate_core
        self.ua_lazy_alpha = float(os.environ.get("PISEG_UA_LAZY_ALPHA", "0.2"))

        # uncertainty 计算方式：
        #   margin  : 1 - (top1_prob - top2_prob)
        #   entropy : normalized entropy
        self.ua_lazy_unc_mode = os.environ.get(
            "PISEG_UA_LAZY_UNC",
            "margin",
        ).lower()

        # LazyScore 计算参数
        self.ua_lazy_low_ratio = float(os.environ.get("PISEG_UA_LAZY_LOW_RATIO", "0.25"))
        self.ua_lazy_topk_ratio = float(os.environ.get("PISEG_UA_LAZY_TOPK_RATIO", "0.15"))

        # 是否打印调试日志。多卡训练时日志会很多，正式训练建议关掉。
        self.ua_lazy_verbose = os.environ.get("PISEG_UA_LAZY_VERBOSE", "0") == "1"

        valid_gate_types = {"none", "unc", "lazy", "unc_lazy"}
        if self.ua_lazy_gate_type not in valid_gate_types:
            raise ValueError(
                f"Unknown PISEG_UA_LAZY_GATE_TYPE={self.ua_lazy_gate_type}. "
                f"Valid choices: {sorted(valid_gate_types)}"
            )

        # 只打印一次初始化配置，用来确认环境变量是否真的传进来了
        print(
            "[UA-Lazy-Config] "
            f"enabled={self.ua_lazy_gate}, "
            f"type={self.ua_lazy_gate_type}, "
            f"alpha={self.ua_lazy_alpha}, "
            f"unc_mode={self.ua_lazy_unc_mode}, "
            f"low_ratio={self.ua_lazy_low_ratio}, "
            f"topk_ratio={self.ua_lazy_topk_ratio}, "
            f"verbose={self.ua_lazy_verbose}",
            flush=True,
        )

        valid_unc_modes = {"margin", "entropy"}
        if self.ua_lazy_unc_mode not in valid_unc_modes:
            raise ValueError(
                f"Unknown PISEG_UA_LAZY_UNC={self.ua_lazy_unc_mode}. "
                f"Valid choices: {sorted(valid_unc_modes)}"
            )

        self.lazy_vis = os.environ.get("PISEG_LAZY_VIS", "0") == "1"
        self.lazy_vis_dir = os.environ.get(
            "PISEG_LAZY_VIS_DIR",
            os.path.join(vis_save_dir, "lazy_vis_dump"),
        )
        self.lazy_vis_limit = int(os.environ.get("PISEG_LAZY_VIS_LIMIT", "200"))
        self.lazy_vis_count = 0    

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)
        corr = torch.einsum("bchw, btpc -> bpthw", img_feats, text_feats)
        return corr  

    @torch.no_grad()
    def _norm01_spatial(self, x, eps=1e-6):
        """
        x: (B, 1, H, W) or (B, H, W)
        """
        if x.dim() == 3:
            x = x[:, None]

        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)


    @torch.no_grad()
    def _compute_cost_uncertainty(self, image_feature, text):
        """
        从 raw image-text cost map 计算空间不确定性。

        Args:
            image_feature: Tensor, shape (B, C, H, W)
            text: Tensor, shape (B, T, P, C)

        Returns:
            uncertainty: Tensor, shape (B, 1, H, W), normalized to [0, 1]
        """
        # self.correlation(image_feature, text): (B, P, T, H, W)
        # mean over prompt dim -> (B, T, H, W)
        corr = self.correlation(image_feature, text).mean(dim=1)

        if corr.shape[1] <= 1:
            return torch.zeros(
                corr.shape[0],
                1,
                corr.shape[-2],
                corr.shape[-1],
                device=corr.device,
                dtype=corr.dtype,
            )

        prob = corr.float().softmax(dim=1)

        if self.ua_lazy_unc_mode == "entropy":
            # normalized entropy, high means uncertain
            eps = 1e-6
            entropy = -(prob * (prob + eps).log()).sum(dim=1, keepdim=True)
            entropy = entropy / torch.log(
                torch.tensor(float(prob.shape[1]), device=prob.device)
            )
            uncertainty = entropy

        elif self.ua_lazy_unc_mode == "margin":
            # margin uncertainty, high means top1 and top2 are close
            top2 = prob.topk(k=2, dim=1).values
            margin = top2[:, 0:1] - top2[:, 1:2]
            uncertainty = 1.0 - margin

        else:
            raise ValueError(
                f"Unknown PISEG_UA_LAZY_UNC={self.ua_lazy_unc_mode}. "
                "Valid choices: margin, entropy"
            )

        return self._norm01_spatial(uncertainty)

    def _compute_cls_logits_for_final_out(self, image_cls_feature, text):
        """
        Compute CLS-text logits for final-logit correction.

        Args:
            image_cls_feature: Tensor, shape (B, C)
            text: Tensor, shape (B, T, P, C)

        Returns:
            cls_logits: Tensor, shape (B, T)
        """
        image_cls_feature = F.normalize(image_cls_feature, dim=-1)
        text = F.normalize(text, dim=-1)

        # (B, C) x (B, T, P, C) -> (B, T, P)
        cls_logits_prompt = torch.einsum(
            "bc,btpc->btp",
            image_cls_feature,
            text,
        )

        # SegEarth-OV 的 query_features 是 prompt ensemble 后的类别特征；
        # Pi-Seg 这里还有 prompt 维 P，所以用 mean(P) 更接近 SegEarth-OV。
        cls_logits = cls_logits_prompt.mean(dim=-1)  # (B, T)

        return cls_logits

    @torch.no_grad()
    def _build_ua_lazy_gate_map(self, image_feature, text):
        """
        构造用于 Image-SPM variance 的 gate map。

        Args:
            image_feature: Tensor, shape (B, C, H, W)
            text: Tensor, shape (B, T, P, C)

        Returns:
            gate_map: Tensor, shape (B, 1, H, W)
            debug_dict: dict
        """
        B, C, H, W = image_feature.shape

        # none 模式：显式 baseline，用于确认改代码后仍可跑原始行为。
        if self.ua_lazy_gate_type == "none":
            gate_map = torch.ones(
                B, 1, H, W,
                device=image_feature.device,
                dtype=image_feature.dtype,
            )
            debug_dict = {
                "lazy_mean": 0.0,
                "unc_mean": 0.0,
                "gate_mean": 1.0,
                "gate_max": 1.0,
            }
            return gate_map, debug_dict

        lazy_score = compute_lazy_score(
            image_feature,
            low_ratio=self.ua_lazy_low_ratio,
            topk_ratio=self.ua_lazy_topk_ratio,
        ).to(device=image_feature.device, dtype=image_feature.dtype)

        uncertainty = self._compute_cost_uncertainty(
            image_feature,
            text,
        ).to(device=image_feature.device, dtype=image_feature.dtype)

        if self.ua_lazy_gate_type == "unc":
            gate_core = uncertainty
        elif self.ua_lazy_gate_type == "lazy":
            gate_core = lazy_score
        elif self.ua_lazy_gate_type == "unc_lazy":
            gate_core = uncertainty * lazy_score
        else:
            raise ValueError(
                f"Unknown PISEG_UA_LAZY_GATE_TYPE={self.ua_lazy_gate_type}"
            )

        # gate range: [1, 1 + alpha]，只放大扰动方差，不反向抑制原 Image-SPM。
        gate_map = 1.0 + self.ua_lazy_alpha * gate_core
        gate_map = gate_map.clamp(min=1.0, max=1.0 + self.ua_lazy_alpha)

        debug_dict = {
            "lazy_mean": float(lazy_score.mean().item()),
            "unc_mean": float(uncertainty.mean().item()),
            "gate_mean": float(gate_map.mean().item()),
            "gate_max": float(gate_map.max().item()),
        }
        return gate_map, debug_dict

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        # PiNI config
        ret["pini_enabled"] = cfg.MODEL.PINI.ENABLED
        ret["pini_image_vpn_enabled"] = cfg.MODEL.PINI.IMAGE_VPN_ENABLED
        ret["pini_text_vpn_enabled"] = cfg.MODEL.PINI.TEXT_VPN_ENABLED
        ret["pini_reduction"] = cfg.MODEL.PINI.REDUCTION
        ret["pini_text_noise_std"] = cfg.MODEL.PINI.TEXT_NOISE_STD

        # ===== image noise =====
        ret["pini_image_noise_type"] = cfg.MODEL.PINI.IMAGE_NOISE_TYPE
        ret["pini_image_student_t_df"] = cfg.MODEL.PINI.IMAGE_STUDENT_T_DF

        # ===== text noise =====
        ret["pini_text_noise_type"] = cfg.MODEL.PINI.TEXT_NOISE_TYPE
        ret["pini_text_student_t_df"] = cfg.MODEL.PINI.TEXT_STUDENT_T_DF

        ret["is_vis"] = cfg.MODEL.PINI.IS_VIS
        ret["vis_save_dir"] = os.path.join(cfg.OUTPUT_DIR, "vis_noise")

        return ret
        
    @torch.no_grad()
    def _save_lazy_vis_dump(
        self,
        files_name,
        input_images,
        targets,
        class_names,
        image_feature,
        text,
        out,
    ):
        if self.lazy_vis_count >= self.lazy_vis_limit:
            return

        os.makedirs(self.lazy_vis_dir, exist_ok=True)

        # 原始 CLIP image-text cost map
        # image_feature: (B, C, H, W)
        # text: (B, T, P, C)
        # corr: (B, P, T, H, W)
        corr = self.correlation(image_feature, text).mean(dim=1)  # (B, T, H, W)

        # LazyScore: (B, 1, H, W)
        lazy_score = compute_lazy_score(image_feature)

        # Pi-Seg final prediction logits -> score
        pred = out.sigmoid()  # (B, T, H_out, W_out)

        B = image_feature.shape[0]

        for b in range(B):
            if self.lazy_vis_count >= self.lazy_vis_limit:
                break

            if files_name is not None:
                base = os.path.basename(str(files_name[b]))
            else:
                base = f"sample_{self.lazy_vis_count}.png"

            base = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
            save_path = os.path.join(
                self.lazy_vis_dir,
                base + f"_lazy_{self.lazy_vis_count:05d}.pt",
            )

            item = {
                "file_name": base,
                "image": input_images[b].detach().cpu()
                if input_images is not None
                else None,
                "target": targets[b].detach().cpu()
                if targets is not None
                else None,
                "corr": corr[b].detach().float().cpu(),          # (T, H, W)
                "lazy": lazy_score[b, 0].detach().float().cpu(), # (H, W)
                "pred": pred[b].detach().float().cpu(),          # (T, H_out, W_out)
                "class_names": list(class_names),
            }

            torch.save(item, save_path)
            self.lazy_vis_count += 1

    def forward(
        self, 
        x, 
        vis_guidance, 
        prompt=None, 
        gt_cls=None,
        files_name=None,
        input_images=None,
        targets=None,
        image_cls_feature=None,
    ):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        class_names = self.class_texts if self.training else self.test_class_texts
        class_names = [class_names[c] for c in gt_cls] if gt_cls is not None else class_names
        text = self.get_text_embeds(class_names, self.prompt_templates, self.clip_model, prompt)

        # PiNI: text feature noise (training only)
        if (self.training or self.is_vis) and self.text_vpn is not None:
            text = self.text_vpn(text)


        text = text.repeat(x.shape[0], 1, 1, 1)
        image_feature_before_noise = x.detach()
        # PiNI: image feature noise guided by text cross-attention (training only)
        im_noise = None
        ua_lazy_gate_map = None

        if (self.training or self.is_vis) and self.image_vpn is not None:
            B, C, H, W = x.shape

            spatial_feat = rearrange(x, 'B C H W -> B (H W) C')

            # Mean-pool text across prompts: (B, T, P, C) -> (T, C)
            text_for_attn = text[0].mean(dim=1).detach()

            mu, variance = self.image_vpn(spatial_feat.detach(), text_for_attn)

            # ========================================================
            # Uncertainty-aware Lazy gate for Image-SPM variance
            # ========================================================
            if self.ua_lazy_gate:
                with torch.no_grad():
                    ua_lazy_gate_map, ua_debug = self._build_ua_lazy_gate_map(
                        image_feature_before_noise,
                        text,
                    )

                    gate_flat = rearrange(
                        ua_lazy_gate_map,
                        'B 1 H W -> B (H W) 1',
                    )

                variance = variance * gate_flat.to(
                    dtype=variance.dtype,
                    device=variance.device,
                )

                if self.ua_lazy_verbose:
                    print(
                        "[UA-Lazy] "
                        f"type={self.ua_lazy_gate_type}, "
                        f"alpha={self.ua_lazy_alpha:.3f}, "
                        f"unc_mode={self.ua_lazy_unc_mode}, "
                        f"lazy_mean={ua_debug['lazy_mean']:.4f}, "
                        f"unc_mean={ua_debug['unc_mean']:.4f}, "
                        f"gate_mean={ua_debug['gate_mean']:.4f}, "
                        f"gate_max={ua_debug['gate_max']:.4f}",
                        flush=True,
                    )

            im_noise = self.image_vpn.sample(mu, variance)

            x = x + rearrange(im_noise, 'B (H W) C -> B C H W', H=H, W=W)

        image_feature_after_noise = x.detach()
        
        
        corr_before = None
        corr_after = None
        delta_corr = None
        if self.training or self.is_vis:
            corr_before = self.correlation(image_feature_before_noise, text)
            corr_after = self.correlation(image_feature_after_noise, text)
            delta_corr = corr_after - corr_before
            self.cached_corr_before = corr_before.detach()
            self.cached_corr_after = corr_after.detach()
            self.cached_delta_corr = delta_corr.detach()
        else:
            self.cached_corr_before = None
            self.cached_corr_after = None
            self.cached_delta_corr = None
        
        if self.is_vis:
            self.visualizer.save_noise_visuals(files_name, None, im_noise)
            self.visualizer.save_visual_grid(
                files_name=files_name,
                input_images=input_images,
                image_feature=image_feature_before_noise,
                image_noise=im_noise,
                noisy_image_feature=image_feature_after_noise,
            )
            
            # ==== GT
            self.visualizer.save_corr_visuals_by_gt(
                files_name=files_name,
                class_names=class_names,
                input_images=input_images,
                corr_before=corr_before,
                delta_corr=delta_corr,
                corr_after=corr_after,
                targets=targets,
                ignore_value=255,
                prompt_reduce="mean",
                alpha=0.60,
            )
            # ===== TOPK
            # self.visualizer.save_corr_visuals(
            #     files_name=files_name,
            #     class_names=class_names,
            #     input_images=input_images,
            #     corr_before=corr_before,
            #     delta_corr=delta_corr,
            #     corr_after=corr_after,
            #     targets=targets,
            #     ignore_value=255,
            #     prompt_reduce="mean",
            #     topk=8,
            #     score_reduce="mean",
            #     alpha=0.60,
            #     use_gt_only=True,
            #     gt_select_mode="largest",
            # )

            if im_noise is not None:
                noise_ratio = (
                    im_noise.detach().norm(dim=1).mean()
                    / (image_feature_before_noise.detach().norm(dim=1).mean() + 1e-6)
                ).item()
                delta_ratio = (
                    delta_corr.detach().abs().mean()
                    / (corr_before.detach().abs().mean() + 1e-6)
                ).item()
                print(f"[VIS] noise_ratio={noise_ratio:.6f}, delta_ratio={delta_ratio:.6f}")


        # ------------------------------------------------------------
        # Original Pi-Seg Aggregator forward.
        # For variant A, do NOT pass CLS to Aggregator.
        # We apply CLS subtraction after final logits are produced.
        # ------------------------------------------------------------
        out = self.transformer(x, text, vis)

        # ------------------------------------------------------------
        # SegEarth-OV style Global Bias Alleviation at final logits
        # ------------------------------------------------------------
        use_segearth_gba = (
            self.segearth_gba
            and image_cls_feature is not None
            and ((not self.training) or self.segearth_apply_train)
        )

        if use_segearth_gba:
            cls_logits = self._compute_cls_logits_for_final_out(
                image_cls_feature=image_cls_feature,
                text=text,
            )  # (B, T)

            if cls_logits.shape[0] != out.shape[0] or cls_logits.shape[1] != out.shape[1]:
                raise RuntimeError(
                    "[SegEarth-GBA-Final] Shape mismatch: "
                    f"cls_logits={tuple(cls_logits.shape)}, out={tuple(out.shape)}"
                )

            out = out + self.segearth_gba_lambda * cls_logits[:, :, None, None].to(
                device=out.device,
                dtype=out.dtype,
            )

            if self.segearth_verbose:
                print(
                    "[SegEarth-GBA-Final] "
                    f"training={self.training}, "
                    f"lambda={self.segearth_gba_lambda}, "
                    f"out_shape={tuple(out.shape)}, "
                    f"cls_logits_shape={tuple(cls_logits.shape)}, "
                    f"cls_min={cls_logits.min().item():.4f}, "
                    f"cls_max={cls_logits.max().item():.4f}, "
                    f"cls_mean={cls_logits.mean().item():.4f}",
                    flush=True,
                )

        if (not self.training) and self.lazy_vis:
            self._save_lazy_vis_dump(
                files_name=files_name,
                input_images=input_images,
                targets=targets,
                class_names=class_names,
                image_feature=image_feature_before_noise,
                text=text,
                out=out,
            )

        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings


# ============================================================
# CATSeg Head (Detectron2)
# ============================================================

@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int,
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution
        self.predictor = transformer_predictor

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return dict(
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            feature_resolution=cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            transformer_predictor=CATSegPredictor(cfg),
        )

    def forward(
        self, 
        features, 
        guidance_features, 
        prompt=None, 
        gt_cls=None, 
        files_name=None, 
        input_images=None,
        targets=None,
    ):
        # features: (B, 1 + H*W, C)
        # features[:, 0, :] is CLIP image CLS token
        # features[:, 1:, :] are patch tokens
        image_cls_feature = features[:, 0, :]

        img_feat = rearrange(
            features[:, 1:, :],
            "b (h w) c -> b c h w",
            h=self.feature_resolution[0],
            w=self.feature_resolution[1],
        )

        return self.predictor(
            img_feat,
            guidance_features,
            prompt,
            gt_cls,
            files_name=files_name,
            input_images=input_images,
            targets=targets,
            image_cls_feature=image_cls_feature,
        )