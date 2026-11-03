"""Frozen VLM wrappers for multi-turn observation histories (greedy decoding).

Every protocol presents its observations as alternating turns:

    user:      "<label> <opaque ID>:" + image
    assistant: "Observation stored."

followed by one final user turn with the protocol's state request. InternVL2.5
uses its native ``model.chat`` history interface with one 448x448 tile per
observation; Qwen2.5-VL, LLaVA-OneVision, and Idefics2 use their Hugging Face
chat templates.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .common import resolve

ACK = "Observation stored."


def _patch_idefics2_position_embedding(torch) -> None:
    from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionEmbeddings

    def forward(module, pixel_values, patch_attention_mask):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        embeddings = module.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        max_h, max_w = max_im_h // module.patch_size, max_im_w // module.patch_size
        boundaries = torch.arange(
            1 / module.num_patches_per_side, 1.0, 1 / module.num_patches_per_side,
            device=pixel_values.device, dtype=pixel_values.dtype)
        position_ids = torch.zeros((batch_size, max_h * max_w),
                                   device=pixel_values.device, dtype=torch.long)
        for batch_idx, mask in enumerate(patch_attention_mask):
            nb_h, nb_w = mask[:, 0].sum(), mask[0].sum()
            hs = torch.arange(nb_h, device=pixel_values.device,
                              dtype=pixel_values.dtype) / nb_h * (1 - 1e-6)
            ws = torch.arange(nb_w, device=pixel_values.device,
                              dtype=pixel_values.dtype) / nb_w * (1 - 1e-6)
            hb = torch.bucketize(hs, boundaries, right=True)
            wb = torch.bucketize(ws, boundaries, right=True)
            position_ids[batch_idx][mask.view(-1)] = (
                hb[:, None] * module.num_patches_per_side + wb).flatten()
        return embeddings + module.position_embedding(position_ids)

    Idefics2VisionEmbeddings.forward = forward


class VLM:
    def __init__(self, spec: dict, device: str):
        import torch
        from transformers import AutoProcessor

        self.torch, self.kind = torch, spec["model_type"]
        self.max_new_tokens = spec["max_new_tokens"]
        path = str(resolve(spec["local_dir"]))
        if self.kind == "internvl2_5":
            from transformers import AutoModel, AutoTokenizer
            self.image_size = spec["image_size"]
            self.model = AutoModel.from_pretrained(
                path, torch_dtype=torch.bfloat16, use_flash_attn=False,
                trust_remote_code=True, local_files_only=True).eval().to(device)
            if not hasattr(self.model.language_model, "generate"):
                from transformers import GenerationMixin
                original = self.model.language_model.__class__
                self.model.language_model.__class__ = type(
                    f"{original.__name__}GenerationCompat", (original, GenerationMixin), {})
            if getattr(self.model.language_model, "generation_config", None) is None:
                from transformers import GenerationConfig
                self.model.language_model.generation_config = \
                    GenerationConfig.from_model_config(self.model.language_model.config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, use_fast=False, local_files_only=True)
            return
        processor_kwargs = {"use_fast": False, "local_files_only": True}
        if self.kind == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
            processor_kwargs.update(min_pixels=96 * 28 * 28, max_pixels=256 * 28 * 28)
        elif self.kind == "llava_onevision":
            from transformers import LlavaOnevisionForConditionalGeneration as cls
        elif self.kind == "idefics2":
            from transformers import Idefics2ForConditionalGeneration as cls
            _patch_idefics2_position_embedding(torch)
        else:
            raise ValueError(f"unsupported model type: {self.kind}")
        self.processor = AutoProcessor.from_pretrained(path, **processor_kwargs)
        self.model = cls.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="sdpa", local_files_only=True).eval()

    def _internvl_tensor(self, path):
        image = Image.open(resolve(path)).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1)
        mean = self.torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((tensor - mean) / std).to(self.torch.bfloat16)

    def history_chat(self, system: str, turns: list[tuple[str, str]], final: str) -> str:
        """Generate after a history of ``(label, image_path)`` observation turns.

        InternVL's ``chat`` interface has no system turn here, matching the frozen runs.
        """
        if self.kind == "internvl2_5":
            tensors = [self._internvl_tensor(path) for _, path in turns]
            history = [(f"{label} <image>", ACK) for label, _ in turns]
            pixels = self.torch.stack(tensors).to(self.model.device)
            return self.model.chat(
                self.tokenizer, pixels, final,
                {"max_new_tokens": self.max_new_tokens, "do_sample": False},
                history=history, num_patches_list=[1] * len(tensors),
                verbose=False).strip()
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        images = []
        for label, path in turns:
            images.append(Image.open(resolve(path)).convert("RGB"))
            messages.append({"role": "user", "content": [
                {"type": "text", "text": label}, {"type": "image"}]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": ACK}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": final}]})
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images,
                                return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=False, use_cache=True)
        return self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
