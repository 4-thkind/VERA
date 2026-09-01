"""
VERA Attention Extractor
Extracts cross-modal attention from LLaVA-style vision-language models.

Key Insight:
LLaVA-style models (CheXagent-2, LLaVA-Med) project image patches into the
LLM's embedding space as tokens. The LLM's self-attention over these "image tokens"
serves as our cross-attention proxy. We extract attention[text_tokens → image_tokens]
and reshape the image-token dimension into a spatial grid matching the vision encoder's
patch layout.
"""
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class AttentionExtractor:
    """
    Extract cross-modal attention from LLaVA-style VLMs.
    
    Works with CheXagent-2-3b (SigLIP + Phi-2) and LLaVA-Med (CLIP + Mistral-7B).
    Both use MLP projection to embed image patches as tokens in the LLM sequence.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        patch_grid: Tuple[int, int] = (24, 24),
        num_layers_to_use: int = 4,
        device: str = "cuda",
    ):
        """
        Args:
            model: HuggingFace VLM model
            tokenizer: HuggingFace tokenizer
            image_processor: Image processor for the vision encoder
            patch_grid: (H, W) of the vision encoder's patch grid
            num_layers_to_use: Number of attention layers from the end to average
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patch_grid = patch_grid
        self.num_layers_to_use = num_layers_to_use
        self.device = device
        self._attention_store = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on all self-attention layers of the LLM."""
        self._clear_hooks()
        self._attention_store = {}

        # Find the language model backbone
        lm_model = None
        for name in ["language_model", "model", "lm_head"]:
            if hasattr(self.model, name):
                lm_model = getattr(self.model, name)
                break

        if lm_model is None:
            lm_model = self.model

        # Find attention layers
        layer_count = 0
        for name, module in lm_model.named_modules():
            # Common attention module names across different architectures
            if any(attn_name in name for attn_name in ["self_attn", "attention", "attn"]):
                if hasattr(module, "forward"):
                    # Only hook into the attention module itself, not sub-modules
                    if "self_attn" in name and "." not in name.split("self_attn")[-1]:
                        layer_name = f"layer_{layer_count}"

                        def hook_fn(module, input, output, name=layer_name):
                            # output format varies by model:
                            # Phi-2: (attn_output, attn_weights, past_key_value)
                            # Mistral: (attn_output, attn_weights, past_key_value)
                            if isinstance(output, tuple) and len(output) >= 2:
                                attn_weights = output[1]
                                if attn_weights is not None:
                                    self._attention_store[name] = attn_weights.detach().cpu()

                        hook = module.register_forward_hook(hook_fn)
                        self._hooks.append(hook)
                        layer_count += 1

        print(f"  Registered {layer_count} attention hooks")
        return layer_count

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_store = {}

    def _find_image_token_positions(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor = None,
    ) -> Tuple[int, int]:
        """
        Find the start and end positions of image tokens in the input sequence.
        
        In CheXagent-2-3b:
        Image tokens are placed between tokenizer.img_start_id (<|img|>) and
        tokenizer.img_end_id (<|/img|>). The visual encoder projects exactly
        num_patches (e.g., 24x24 = 576) patch features into the sequence.
        
        In LLaVA-style models:
        Image placeholder tokens (e.g. IMAGE_TOKEN_INDEX / -200 or <image>)
        are expanded to num_patches tokens.
        """
        input_ids_list = input_ids[0].tolist()
        num_patches = self.patch_grid[0] * self.patch_grid[1]
        
        # Method 1: CheXagent custom tokenizer (<|img|> ... <|/img|>)
        img_start_id = getattr(self.tokenizer, "img_start_id", None)
        img_end_id = getattr(self.tokenizer, "img_end_id", None)
        
        if img_start_id is not None and img_start_id in input_ids_list:
            start_idx = input_ids_list.index(img_start_id) + 1
            if img_end_id is not None and img_end_id in input_ids_list:
                end_idx = input_ids_list.index(img_end_id)
                # The model places the visual embeddings starting right after img_start_id
                # If there are pad tokens up to img_end_id, the actual patch count is num_patches
                if (end_idx - start_idx) > num_patches:
                    end_idx = start_idx + num_patches
            else:
                end_idx = min(start_idx + num_patches, len(input_ids_list))
            print(f"  [DEBUG] Found CheXagent image token bounds: [{start_idx}:{end_idx}] (count={end_idx - start_idx})")
            return start_idx, end_idx

        # Method 2: Look for image_token_id in config (for standard LLaVA / LLaVA-Med)
        image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_id", None)

        if image_token_id is not None and image_token_id in input_ids_list:
            positions = [i for i, t in enumerate(input_ids_list) if t == image_token_id]
            if positions:
                start_idx = positions[0]
                end_idx = positions[-1] + 1
                if (end_idx - start_idx) == 1 and num_patches > 1:
                    end_idx = start_idx + num_patches
                print(f"  [DEBUG] Found LLaVA image token bounds: [{start_idx}:{end_idx}] (count={end_idx - start_idx})")
                return start_idx, end_idx

        # Fallback: estimate from number of patches
        prompt_length = min(10, max(1, len(input_ids_list) // 4))
        img_start = prompt_length
        img_end = min(prompt_length + num_patches, len(input_ids_list))
        print(f"  [DEBUG] Fallback image token bounds: [{img_start}:{img_end}] (count={img_end - img_start})")
        return img_start, img_end

    def extract_attention(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> Dict:
        """
        Generate a report and extract cross-modal attention maps.
        
        Args:
            image: PIL Image
            prompt: Text prompt for report generation
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict with:
            - generated_text: str
            - attention_maps: np.ndarray of shape [num_generated_tokens, H_patches, W_patches]
            - raw_attention_shape: original attention tensor shape info
        """
        # Register hooks to capture attention
        num_hooks = self._register_hooks()

        try:
            # Process inputs
            if hasattr(self.tokenizer, "from_list_format"):
                # Fix Stanford's bug where pos_embed is float32, which casts images to float32
                if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
                    visual = self.model.model.visual
                    
                    # 1. Materialize pos_embed if it was left on the meta device
                    if hasattr(visual, "pos_embed") and visual.pos_embed.device.type == "meta":
                        import sys
                        module = sys.modules[visual.__module__]
                        width = visual.model.config.hidden_size
                        grid_size = visual.grid_size[0]
                        pos_embed_np = module.get_2d_sincos_pos_embed(width, grid_size)
                        # Replace the meta tensor entirely with a real parameter
                        visual.pos_embed = torch.nn.Parameter(
                            torch.from_numpy(pos_embed_np).to(device=self.device, dtype=torch.float16), 
                            requires_grad=False
                        )

                    # 2. Reload the ENTIRE vision encoder if accelerate turned it into a ghost
                    if hasattr(visual, "model") and next(visual.model.parameters()).device.type == "meta":
                        from transformers import AutoModel
                        print("    [FIX] Reloading SigLIP vision model to replace meta tensors...")
                        real_vision_model = AutoModel.from_pretrained(
                            "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
                        ).vision_model
                        visual.model = real_vision_model.to(device=self.device, dtype=torch.float16)

                    # 3. Fix precision bugs on real tensors
                    for param in visual.parameters():
                        if param.dtype == torch.float32 and param.device.type != "meta":
                            param.data = param.data.to(torch.float16)
                import tempfile
                tmp_img = tempfile.mktemp(suffix=".png")
                image.save(tmp_img)
                query = self.tokenizer.from_list_format([{"image": tmp_img}, {"text": prompt}])
                conv = [
                    {"from": "system", "value": "You are a helpful assistant."},
                    {"from": "human", "value": query}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                text_inputs = self.tokenizer(
                    text=formatted_prompt, return_tensors="pt", padding=True
                ).to(self.device)
                pixel_values = None
            else:
                if self.image_processor is not None:
                    pixel_values = self.image_processor(
                        images=image, return_tensors="pt"
                    ).pixel_values.to(self.device, dtype=self.model.dtype)
                else:
                    pixel_values = None
                text_inputs = self.tokenizer(
                    text=prompt, return_tensors="pt", padding=True
                ).to(self.device)

            # Build model inputs
            model_inputs = {
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask,
            }
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values

            print("  [DEBUG] Calling model.generate()...")
            # Generate with attention output
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

            # Decode generated text
            generated_ids = outputs.sequences[0]
            input_length = model_inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                generated_ids[input_length:], skip_special_tokens=True
            )

            # Process attention maps
            attention_maps = self._process_attention(
                outputs, model_inputs["input_ids"], pixel_values
            )

            return {
                "generated_text": generated_text.strip(),
                "attention_maps": attention_maps,
                "num_generated_tokens": len(generated_ids) - input_length,
                "input_length": input_length,
            }

        finally:
            self._clear_hooks()

    def _process_attention(
        self,
        generate_outputs,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor = None,
    ) -> np.ndarray:
        """
        Process raw attention outputs into spatial attention maps.
        
        Extracts attention from generated tokens to image tokens,
        averages over heads and last N layers, and reshapes to
        spatial grid.
        
        Returns:
            np.ndarray of shape [num_generated_tokens, H_patches, W_patches]
        """
        H_patches, W_patches = self.patch_grid
        num_image_patches = H_patches * W_patches

        # Get image token positions
        img_start, img_end = self._find_image_token_positions(input_ids, pixel_values)
        actual_num_patches = img_end - img_start

        # Method 1: Use generate_outputs.attentions (per-step attentions)
        if hasattr(generate_outputs, 'attentions') and generate_outputs.attentions is not None:
            attention_maps = self._process_generate_attentions(
                generate_outputs.attentions,
                img_start, img_end,
                num_image_patches
            )
            if attention_maps is not None:
                return attention_maps

        # Method 2: Use hook-captured attentions
        if self._attention_store:
            attention_maps = self._process_hook_attentions(
                img_start, img_end, num_image_patches
            )
            if attention_maps is not None:
                return attention_maps

        # Fallback: return uniform attention
        print("  [WARN] Could not extract attention maps. Using uniform attention.")
        return np.ones((1, H_patches, W_patches), dtype=np.float32) / num_image_patches

    def _process_generate_attentions(
        self,
        attentions,
        img_start: int,
        img_end: int,
        num_image_patches: int,
    ) -> Optional[np.ndarray]:
        """Process attentions from model.generate() output."""
        H_patches, W_patches = self.patch_grid
        
        try:
            all_step_maps = []

            # attentions is a tuple of per-step attention tuples
            # Each step: tuple of [num_layers] tensors
            # Each tensor: [batch, num_heads, seq_len_at_step, seq_len_at_step]
            for step_idx, step_attentions in enumerate(attentions):
                if step_attentions is None:
                    continue

                # Use last N layers
                n_layers = min(self.num_layers_to_use, len(step_attentions))
                layer_maps = []

                for layer_attn in step_attentions[-n_layers:]:
                    if layer_attn is None:
                        continue

                    attn = layer_attn.detach().cpu().numpy()
                    # Shape: [batch, heads, query_len, key_len]

                    # For generation step: query = last token (newly generated)
                    # We want attention from this token to image patches
                    if attn.shape[2] == 1:
                        # Single query token (typical for generation steps after first)
                        attn_to_image = attn[0, :, 0, img_start:img_end]
                    else:
                        # Multiple query tokens (first step / prefill)
                        attn_to_image = attn[0, :, -1, img_start:img_end]

                    # Average over heads: [num_patches]
                    attn_avg = attn_to_image.mean(axis=0)
                    layer_maps.append(attn_avg)

                if layer_maps:
                    # Average over layers
                    step_map = np.mean(layer_maps, axis=0)
                    # Normalize
                    step_sum = step_map.sum()
                    if step_sum > 0:
                        step_map = step_map / step_sum

                    # Reshape to spatial grid
                    if len(step_map) == num_image_patches:
                        step_map_2d = step_map.reshape(H_patches, W_patches)
                    else:
                        # Interpolate if patch count doesn't match
                        from scipy.ndimage import zoom
                        side = int(np.sqrt(len(step_map)))
                        if side * side == len(step_map):
                            step_map_2d = step_map.reshape(side, side)
                            step_map_2d = zoom(
                                step_map_2d,
                                (H_patches / side, W_patches / side),
                                order=1
                            )
                        else:
                            step_map_2d = np.ones((H_patches, W_patches)) / num_image_patches

                    all_step_maps.append(step_map_2d)

            if all_step_maps:
                return np.array(all_step_maps, dtype=np.float32)

        except Exception as e:
            print(f"  [WARN] Error processing generate attentions: {e}")

        return None

    def _process_hook_attentions(
        self,
        img_start: int,
        img_end: int,
        num_image_patches: int,
    ) -> Optional[np.ndarray]:
        """Process attentions captured via forward hooks."""
        H_patches, W_patches = self.patch_grid

        try:
            # Sort layers by name
            layer_names = sorted(self._attention_store.keys())
            n_layers = min(self.num_layers_to_use, len(layer_names))
            use_layers = layer_names[-n_layers:]

            layer_maps = []
            for layer_name in use_layers:
                attn = self._attention_store[layer_name].numpy()
                # Shape: [batch, heads, seq_len, seq_len]

                # Last generated token's attention to image patches
                attn_to_image = attn[0, :, -1, img_start:img_end]
                attn_avg = attn_to_image.mean(axis=0)
                layer_maps.append(attn_avg)

            if layer_maps:
                combined = np.mean(layer_maps, axis=0)
                combined_sum = combined.sum()
                if combined_sum > 0:
                    combined = combined / combined_sum

                if len(combined) == num_image_patches:
                    return combined.reshape(1, H_patches, W_patches).astype(np.float32)

        except Exception as e:
            print(f"  [WARN] Error processing hook attentions: {e}")

        return None


# ============================================================
# Batch Processing
# ============================================================

def process_batch(
    extractor: AttentionExtractor,
    data: List[Dict],
    output_dir: str,
    prompt: str,
    max_new_tokens: int = 256,
    save_every: int = 10,
) -> List[Dict]:
    """
    Process a batch of images through the attention extractor.
    
    Args:
        extractor: AttentionExtractor instance
        data: List of dicts with 'image_id' and 'image_path'
        output_dir: Directory to save attention maps
        prompt: Generation prompt
        max_new_tokens: Max tokens to generate
        save_every: Save checkpoint every N images
    
    Returns:
        List of result dicts with generated reports and attention file paths
    """
    from PIL import Image
    from tqdm import tqdm
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, entry in enumerate(tqdm(data, desc="Model inference")):
        image_id = entry["image_id"]
        image_path = entry["image_path"]

        # Check if already processed
        attn_path = output_dir / f"{image_id}_attention.npz"
        if attn_path.exists():
            # Load existing result
            try:
                meta_path = output_dir / f"{image_id}_meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    results.append(meta)
                    continue
            except Exception:
                pass

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Extract attention
            result = extractor.extract_attention(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            # Save attention maps as compressed numpy
            np.savez_compressed(
                str(attn_path),
                attention_maps=result["attention_maps"],
            )

            # Save metadata
            meta = {
                "image_id": image_id,
                "image_path": str(image_path),
                "generated_report": result["generated_text"],
                "attention_map_path": str(attn_path),
                "num_generated_tokens": result["num_generated_tokens"],
            }
            meta_path = output_dir / f"{image_id}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)

        except Exception as e:
            print(f"\n  [ERROR] Failed on {image_id}: {e}")
            results.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "error": str(e),
            })

        # Periodic checkpoint
        if (idx + 1) % save_every == 0:
            checkpoint_path = output_dir / "results_checkpoint.json"
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    final_path = output_dir / "inference_results.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {len(results)} results to {final_path}")

    return results


# ============================================================
# Model Loading Helpers
# ============================================================

def load_chexagent(
    model_id: str = "StanfordAIMI/CheXagent-2-3b",
    hf_token: str = None,
    device: str = "cuda",
    load_in_4bit: bool = False,
) -> Tuple:
    """
    Load CheXagent-2-3b model with attention output support.
    
    Returns:
        (model, tokenizer, image_processor)
    """
    import tokenizers
    tokenizers.__version__ = "0.19.1"
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    print(f"  Loading {model_id}...")

    # Quantization config for limited VRAM
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load directly as AutoModel since CheXagent uses custom architecture
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=hf_token,
        trust_remote_code=True,
        attn_implementation="eager",
        quantization_config=quantization_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )

    # Load image processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
        image_processor = processor.image_processor if hasattr(processor, 'image_processor') else processor
    except Exception:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )

    model.eval()
    print(f"  Model loaded on {device}")

    return model, tokenizer, image_processor


def load_llava_med(
    model_id: str = "microsoft/llava-med-v1.5-mistral-7b",
    hf_token: str = None,
    device: str = "cuda",
    load_in_4bit: bool = True,
) -> Tuple:
    """
    Load LLaVA-Med model with attention output support.
    
    Returns:
        (model, tokenizer, image_processor)
    """
    from transformers import (
        LlavaForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    print(f"  Loading {model_id}...")

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=hf_token,
        attn_implementation="eager",
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    image_processor = processor.image_processor

    model.eval()
    print(f"  Model loaded on {device}")

    return model, tokenizer, image_processor
