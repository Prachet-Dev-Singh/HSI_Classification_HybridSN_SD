# model_swin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath, to_2tuple, trunc_normal_ # Use new import path
from timm.models.swin_transformer import window_partition, window_reverse # Helper functions from timm
# We will create our own SwinTransformerBlockCustom and BasicLayerCustom
# to facilitate outputting intermediate features for self-distillation.

# --- Configuration Import (Ensure these are defined in your config.py) ---
from config import (
    # From Step 1 config.py
    SWIN_INITIAL_PATCH_EMBED_DIM, SWIN_APE, SWIN_PATCH_NORM,
    SWIN_DEPTHS, SWIN_NUM_HEADS, SWIN_WINDOW_SIZES,
    SWIN_MLP_RATIO, SWIN_QKV_BIAS, SWIN_QK_SCALE, # QK_SCALE can be None
    SWIN_DROP_RATE, SWIN_ATTN_DROP_RATE, SWIN_DROP_PATH_RATE,
    NUM_CLASSES_ACTUAL # This will be passed to the model
)

# --- Auxiliary Head (from previous steps) ---
class AuxHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        # Ensure num_classes is valid, it should be set by data_preprocessing
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"AuxHead: num_classes must be a positive integer, got {num_classes}")
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x): # x is (B, embed_dim) - CLS token output
        return self.fc(x)

# --- Swin-style Patch Embedding (from previous step) ---
class SwinPatchEmbed(nn.Module):
    def __init__(self, img_size_h, img_size_w, sub_patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = (img_size_h, img_size_w)
        self.sub_patch_size = to_2tuple(sub_patch_size)
        self.num_patches_h = img_size_h // self.sub_patch_size[0]
        self.num_patches_w = img_size_w // self.sub_patch_size[1]
        self.num_patches_total = self.num_patches_h * self.num_patches_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.sub_patch_size, stride=self.sub_patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        # ... (same as your Step 1 SwinPatchEmbed forward)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input HSI patch size ({H}*{W}) doesn't match SwinPatchEmbed expected size ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, self.num_patches_h, self.num_patches_w

# --- Swin-style Patch Merging (from previous step) ---
class SwinPatchMerging(nn.Module):
    def __init__(self, input_token_grid_hw, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # ... (same as your Step 1 SwinPatchMerging init) ...
        self.input_H, self.input_W = input_token_grid_hw
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.input_H % 2 != 0 or self.input_W % 2 != 0:
            print(f"WARNING: SwinPatchMerging initialized with odd input token grid H={self.input_H}, W={self.input_W}.")
        self.norm = norm_layer(4 * input_dim)
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
    def forward(self, x_img_tokens, H_in, W_in):
        # ... (same as your Step 1 SwinPatchMerging forward, ensure it handles truncation for odd H/W correctly) ...
        B, L, C = x_img_tokens.shape
        assert L == H_in * W_in, "Input x_img_tokens has wrong number of tokens for given H_in, W_in."
        assert C == self.input_dim, "Input x_img_tokens has wrong channel dimension."
        x_img_tokens = x_img_tokens.view(B, H_in, W_in, C)
        H_out, W_out = H_in // 2, W_in // 2
        if H_out == 0 or W_out == 0:
            return torch.empty(B, 0, self.output_dim, device=x_img_tokens.device), H_out, W_out
        x0 = x_img_tokens[:, 0:2*H_out:2, 0:2*W_out:2, :]
        x1 = x_img_tokens[:, 1:2*H_out:2, 0:2*W_out:2, :]
        x2 = x_img_tokens[:, 0:2*H_out:2, 1:2*W_out:2, :]
        x3 = x_img_tokens[:, 1:2*H_out:2, 1:2*W_out:2, :]
        x0 = x0[:, :H_out, :W_out, :]
        x1 = x1[:, :H_out, :W_out, :]
        x2 = x2[:, :H_out, :W_out, :]
        x3 = x3[:, :H_out, :W_out, :]
        x_merged = torch.cat([x0, x1, x2, x3], -1)
        x_merged = x_merged.view(B, -1, 4 * C)
        x_merged = self.norm(x_merged)
        x_merged = self.reduction(x_merged)
        return x_merged, H_out, W_out

# --- Custom MLP for Swin Block (same as SwinMLP from before) ---
class SwinMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * SWIN_MLP_RATIO) # Use configured MLP_RATIO
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# --- WindowAttention (Adapted from timm.models.swin_transformer.WindowAttention) ---
class WindowAttentionCustom(nn.Module):
    def __init__(self, dim, window_size_hw, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size_h, self.window_size_w = window_size_hw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size_h - 1) * (2 * self.window_size_w - 1), num_heads))

        coords_h = torch.arange(self.window_size_h)
        coords_w = torch.arange(self.window_size_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size_h - 1
        relative_coords[:, :, 1] += self.window_size_w - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None): # x shape: (num_windows*B, N_tokens_in_window, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_h * self.window_size_w, self.window_size_h * self.window_size_w, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None: # SW-MSA mask
            nW = mask.shape[0] # num_windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else: # W-MSA
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- SwinTransformerBlockCustom (Adapted from timm, to return intermediate feature) ---
# model_swin.py
# ... (other parts of the file: imports, AuxHead, SwinPatchEmbed, SwinPatchMerging, SwinMLP, WindowAttentionCustom) ...

# In model_swin.py

# ... (Keep all other imports and class definitions like AuxHead, SwinPatchEmbed, SwinPatchMerging, SwinMLP, WindowAttentionCustom, and the __init__ of SwinTransformerBlockCustom as they were in the previous successful run that fixed the TypeError) ...

class SwinTransformerBlockCustom(nn.Module):
    def __init__(self, dim, input_resolution_hw, num_heads, 
                 window_size=7, # Not used by global MHSA, but kept for signature compatibility
                 shift_size=0,  # Not used by global MHSA
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, # qk_scale not directly used by nn.MultiheadAttention
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_cls_token_in_block=True): # This flag is still relevant
        super().__init__()
        self.dim = dim
        self.input_resolution_h, self.input_resolution_w = input_resolution_hw # For assertions if needed
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_cls_token_in_block = use_cls_token_in_block

        self.norm1 = norm_layer(dim)
        # MODIFIED for Experiment D: Use standard Global MultiheadAttention
        # attn_drop is the dropout for attention weights in nn.MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True, bias=qkv_bias)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity() # For the first residual
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwinMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() # For the second residual
        
        # Debug counter (instance specific)
        self._debug_print_counter = 0
        # SWIN_DEPTHS needs to be accessible for MAX_DEBUG_PRINTS, or pass it, or hardcode a small number
        # For simplicity, let's try to access it globally if config.py was imported by main.py
        try:
            from config import SWIN_DEPTHS as global_SWIN_DEPTHS # Try to import for MAX_DEBUG_PRINTS
            self.MAX_DEBUG_PRINTS = (sum(global_SWIN_DEPTHS) if isinstance(global_SWIN_DEPTHS, list) else 2) + 2
        except ImportError:
            self.MAX_DEBUG_PRINTS = 4 # Fallback if config.SWIN_DEPTHS not found globally

    def forward(self, x_with_cls_maybe, H_current, W_current): # H_current, W_current mostly for assertions now
        B, L, C = x_with_cls_maybe.shape
        
        # Debug print control
        do_debug_print = self._debug_print_counter < self.MAX_DEBUG_PRINTS
        
        if do_debug_print:
            print(f"\n--- GlobalAttnBlock ID: {hex(id(self))} Call: {self._debug_print_counter} ---")
            print(f"  Input (x_with_cls_maybe): shape={x_with_cls_maybe.shape}")
            print(f"    x abs mean: {x_with_cls_maybe.abs().mean().item():.3e}, std: {x_with_cls_maybe.std().item():.3e}")

        # The entire sequence (CLS + image tokens) will go through global attention
        shortcut_full_sequence = x_with_cls_maybe
        
        x_norm1_full = self.norm1(x_with_cls_maybe)
        if do_debug_print: print(f"  x_norm1_full (after norm1): abs mean={x_norm1_full.abs().mean().item():.3e}, std={x_norm1_full.std().item():.3e}")

        # Global MHSA: query, key, value are the same (x_norm1_full)
        # For nn.MultiheadAttention, output is (attn_output, attn_output_weights)
        attn_output_full, _ = self.attn(query=x_norm1_full, key=x_norm1_full, value=x_norm1_full)
        if do_debug_print: print(f"  attn_output_full (Global MHSA output): abs mean={attn_output_full.abs().mean().item():.3e}, std={attn_output_full.std().item():.3e}")

        # First residual connection
        x_after_attn_res = shortcut_full_sequence + self.drop_path1(attn_output_full)
        if do_debug_print: print(f"  x_after_attn_res (after 1st residual): abs mean={x_after_attn_res.abs().mean().item():.3e}, std={x_after_attn_res.std().item():.3e}")

        # MLP Part (applied to the whole sequence after first residual)
        shortcut_for_mlp = x_after_attn_res # Shortcut for the MLP part

        x_norm2_full = self.norm2(x_after_attn_res) # Norm before MLP
        if do_debug_print: print(f"  x_norm2_full (after norm2): abs mean={x_norm2_full.abs().mean().item():.3e}, std={x_norm2_full.std().item():.3e}")
        
        mlp_output_full = self.mlp(x_norm2_full)
        if do_debug_print: print(f"  mlp_output_full (MLP output): abs mean={mlp_output_full.abs().mean().item():.3e}, std={mlp_output_full.std().item():.3e}")
        
        # Second residual connection
        output_final = shortcut_for_mlp + self.drop_path2(mlp_output_full)
        if do_debug_print: print(f"  Output (output_final after 2nd residual): abs mean={output_final.abs().mean().item():.3e}, std={output_final.std().item():.3e}")

        # --- Feature for L2 Hint Loss ---
        features_cls_before_mlp = None
        if self.use_cls_token_in_block:
            # For L2 Hint: We need the CLS token state *before* its final MLP transformation but *after* its attention update and first norm.
            # In this global attention setup, x_norm1_full contains the CLS token after norm1, before global attention.
            # x_after_attn_res contains the CLS token after global attention and first residual.
            # Let's take features_cls_before_mlp from x_norm1_full[:, 0] to be consistent with a pre-activation MLP path.
            features_cls_before_mlp = x_norm1_full[:, 0].detach() # (B, C)
            if do_debug_print: print(f"  Output (features_cls_before_mlp from x_norm1_full[:,0]): shape={features_cls_before_mlp.shape}, abs mean={features_cls_before_mlp.abs().mean().item():.3e}, std={features_cls_before_mlp.std().item():.3e}")
        
        if do_debug_print:
            self._debug_print_counter += 1
            print(f"--- End GlobalAttnBlock Call: {self._debug_print_counter -1} ---")

        return output_final, features_cls_before_mlp


# ... (Rest of model_swin.py: CustomBasicLayer, VisionTransformerSwinHSI, and __main__ block remain unchanged from the full Step 2 code I provided previously) ...
# --- CustomBasicLayer (Sequence of SwinTransformerBlockCustom + Optional PatchMerging) ---
class CustomBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution_hw, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_per_block=None, norm_layer=nn.LayerNorm, downsample_layer=None, use_checkpoint=False,
                 num_classes_for_aux_head=None, use_cls_token_in_blocks=True):
        super().__init__()
        self.dim = dim
        self.input_resolution_h, self.input_resolution_w = input_resolution_hw
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_cls_token_in_blocks = use_cls_token_in_blocks

        self.blocks = nn.ModuleList()
        self.aux_heads_in_layer = nn.ModuleList() # Aux head for each block in this layer

        for i in range(depth):
            # This instantiation uses SwinTransformerBlockCustom.
            # Whether it's the Swin-like version or the Global Attention version (for Exp D)
            # depends on which definition of SwinTransformerBlockCustom is active in the file.
            self.blocks.append(
                SwinTransformerBlockCustom( # This should be your currently active version of the block
                    dim=dim, input_resolution_hw=input_resolution_hw,
                    num_heads=num_heads, window_size=window_size,
                    # For Swin-like: shift_size=0 if (i % 2 == 0) else window_size // 2,
                    # For Global Attention version, shift_size is ignored but still needs a value:
                    shift_size=0 if (i % 2 == 0) else (window_size // 2 if isinstance(window_size, int) else window_size[0] // 2), # Provide a default shift
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path_per_block[i] if isinstance(drop_path_per_block, list) else drop_path_per_block,
                    norm_layer=norm_layer,
                    use_cls_token_in_block=use_cls_token_in_blocks
                )
            )
            if self.use_cls_token_in_blocks:
                if num_classes_for_aux_head is None:
                    raise ValueError("CustomBasicLayer: num_classes_for_aux_head cannot be None if use_cls_token_in_blocks is True.")
                self.aux_heads_in_layer.append(AuxHead(dim, num_classes_for_aux_head))

        self.downsample = downsample_layer # This will be SwinPatchMerging or None
        if self.downsample:
            self.output_dim_after_downsample = self.downsample.output_dim
            # Calculate H, W after potential downsampling for this basic layer's output
            # This assumes downsample is SwinPatchMerging which halves H,W
            self.output_H_after_downsample = self.input_resolution_h // 2
            self.output_W_after_downsample = self.input_resolution_w // 2
        else:
            self.output_dim_after_downsample = dim
            self.output_H_after_downsample = self.input_resolution_h
            self.output_W_after_downsample = self.input_resolution_w


    def forward(self, x, H_in, W_in):
        # x: (B, (1 if cls else 0) + H_in*W_in, C_in)
        
        aux_logits_from_layer = []
        hint_features_from_layer = []
        
        current_x = x
        for i_block, blk in enumerate(self.blocks):
            if self.use_checkpoint and not torch.jit.is_scripting(): # Gradient checkpointing
                # Checkpointing needs H_in, W_in to be passed if blk.forward expects them
                current_x, features_cls_bmlp = checkpoint.checkpoint(blk, current_x, H_in, W_in)
            else:
                current_x, features_cls_bmlp = blk(current_x, H_in, W_in)
            
            if self.use_cls_token_in_blocks:
                # Ensure we are using the correct index for potentially fewer aux_heads if some blocks don't use CLS
                # However, use_cls_token_in_blocks is passed to SwinTransformerBlockCustom and also controls aux_head creation here.
                # So, len(self.aux_heads_in_layer) should match number of blocks if use_cls_token_in_blocks is True for all.
                cls_token_output_for_clf = current_x[:, 0] # (B, C_in)
                aux_logit = self.aux_heads_in_layer[i_block](cls_token_output_for_clf)
                aux_logits_from_layer.append(aux_logit)
                if features_cls_bmlp is not None:
                    hint_features_from_layer.append(features_cls_bmlp)
                else: # Fallback if a block doesn't return a specific hint feature
                    # This might happen if use_cls_token_in_block was false for that block (not current design)
                    # Or if the block's CLS path changed.
                    print(f"Warning: Hint feature (features_cls_bmlp) is None for block {i_block} in CustomBasicLayer. Using detached CLS output as hint.")
                    hint_features_from_layer.append(cls_token_output_for_clf.detach())

        # Patch Merging (if exists for this layer)
        H_out_final, W_out_final = H_in, W_in # Dimensions before potential downsampling by this layer
        x_after_downsample = current_x       # Output of blocks before potential downsampling

        if self.downsample is not None:
            # Separate CLS for merging if used by blocks
            if self.use_cls_token_in_blocks:
                cls_token_before_merge = current_x[:, 0:1, :]    # (B, 1, C_in)
                img_tokens_to_merge = current_x[:, 1:, :]      # (B, H_in*W_in, C_in)
                
                merged_img_tokens, H_out_final, W_out_final = self.downsample(img_tokens_to_merge, H_in, W_in)
                
                # CLS token needs dimension adjustment if merger changes feature dim
                # This requires a dedicated projector, usually defined in VisionTransformerSwinHSI
                # For now, this CustomBasicLayer assumes the CLS token will be handled (projected)
                # by the parent VisionTransformerSwinHSI module after this layer returns.
                # So, we just pass the CLS token as is from before merging, and the main model handles its dim.
                # OR, CustomBasicLayer itself could have a CLS projector if its downsampler changes dim.
                # Let's assume for now the CLS token from *before* merging is what we want to combine
                # with merged_img_tokens, and the parent class will handle any necessary projection.
                # However, to make this layer self-contained regarding output shape:
                if cls_token_before_merge.shape[-1] != self.output_dim_after_downsample:
                    # This implies the CLS token needs projection to match the output dim of merged tokens.
                    # This projector should ideally be part of this layer if it's responsible for outputting consistent dim.
                    # For now, this is a known complexity point. We'll just concatenate assuming parent handles it.
                    # A simple but not learnable way:
                    # cls_token_projected = cls_token_before_merge # This will error if dims mismatch at concat
                    # If we add a projector:
                    if not hasattr(self, 'cls_projector_for_downsample'):
                        if self.dim != self.output_dim_after_downsample: # Only if dim changes
                            self.cls_projector_for_downsample = nn.Linear(self.dim, self.output_dim_after_downsample).to(cls_token_before_merge.device)
                            # Initialize weights for this dynamically added layer
                            trunc_normal_(self.cls_projector_for_downsample.weight, std=.02)
                            if self.cls_projector_for_downsample.bias is not None:
                                nn.init.constant_(self.cls_projector_for_downsample.bias, 0)
                        else: # No projection needed if dims are same
                            self.cls_projector_for_downsample = nn.Identity()
                    
                    cls_token_after_merge = self.cls_projector_for_downsample(cls_token_before_merge)

                else: # Dims already match or no downsampling changed dim
                    cls_token_after_merge = cls_token_before_merge

                x_after_downsample = torch.cat((cls_token_after_merge, merged_img_tokens), dim=1)
            else: # No CLS token involved in blocks
                x_after_downsample, H_out_final, W_out_final = self.downsample(current_x, H_in, W_in)
        
        return x_after_downsample, H_out_final, W_out_final, aux_logits_from_layer, hint_features_from_layer


# --- Main Model: VisionTransformerSwinHSI ---
class VisionTransformerSwinHSI(nn.Module):
    def __init__(self, img_size_hw, sub_patch_size, in_chans, num_classes,
                 initial_embed_dim, depths, num_heads_list, window_sizes,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, use_cls_token=True): # use_cls_token for distillation
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.initial_embed_dim = initial_embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.use_cls_token = use_cls_token # CRITICAL for your self-distillation design
        assert self.use_cls_token, "Self-distillation setup requires use_cls_token=True"

        # 1. Initial Patch Embedding
        self.patch_embed = SwinPatchEmbed(
            img_size_h=img_size_hw[0], img_size_w=img_size_hw[1],
            sub_patch_size=sub_patch_size,
            in_chans=in_chans, embed_dim=initial_embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        initial_grid_h = img_size_hw[0] // to_2tuple(sub_patch_size)[0]
        initial_grid_w = img_size_hw[1] // to_2tuple(sub_patch_size)[1]
        num_initial_img_tokens = initial_grid_h * initial_grid_w

        if self.ape:
            embed_len = num_initial_img_tokens + 1 # For CLS token
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_len, initial_embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, initial_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        # Build Swin stages using CustomBasicLayer
        self.stages = nn.ModuleList()
        self.all_aux_heads_flat = nn.ModuleList() # To collect all aux heads sequentially
        
        current_dim = initial_embed_dim
        current_resolution_hw = (initial_grid_h, initial_grid_w)
        dpr_idx_start = 0

        for i_stage in range(self.num_stages):
            dim_this_stage = initial_embed_dim * (2 ** i_stage) # Or use a list of embed_dims per stage
            # If current_dim from previous stage/merger doesn't match dim_this_stage, a projection is needed.
            # This is handled by SwinPatchMerging output_dim. So, current_dim IS dim_this_stage.

            downsample_layer_for_this_stage = None
            if i_stage < self.num_stages - 1: # If not the last stage, add patch merging
                # Ensure current_resolution_hw is even for standard merging
                if current_resolution_hw[0] % 2 != 0 or current_resolution_hw[1] % 2 != 0:
                    print(f"Stage {i_stage}: Input grid H={current_resolution_hw[0]},W={current_resolution_hw[1]} is odd. "
                          "PatchMerging will truncate. Results might be suboptimal if not handled by padding.")
                
                # Output dim of merger will be input dim for *next* stage's blocks
                next_stage_dim = initial_embed_dim * (2 ** (i_stage + 1))
                downsample_layer_for_this_stage = SwinPatchMerging(
                    input_token_grid_hw=current_resolution_hw,
                    input_dim=current_dim, # dim of tokens *entering* this merger
                    output_dim=next_stage_dim, # dim of tokens *exiting* this merger
                    norm_layer=norm_layer
                )

            stage = CustomBasicLayer(
                dim=current_dim, # Dim for blocks in *this* stage
                input_resolution_hw=current_resolution_hw,
                depth=depths[i_stage],
                num_heads=num_heads_list[i_stage],
                window_size=window_sizes[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path_per_block=dpr[dpr_idx_start : dpr_idx_start + depths[i_stage]],
                norm_layer=norm_layer,
                downsample_layer=downsample_layer_for_this_stage,
                use_checkpoint=use_checkpoint,
                num_classes_for_aux_head=num_classes,
                use_cls_token_in_blocks=self.use_cls_token # Pass this down
            )
            self.stages.append(stage)
            self.all_aux_heads_flat.extend(stage.aux_heads_in_layer) # Collect aux heads

            dpr_idx_start += depths[i_stage]
            if downsample_layer_for_this_stage:
                current_dim = downsample_layer_for_this_stage.output_dim
                current_resolution_hw = (stage.output_H_after_downsample, stage.output_W_after_downsample)
        
        # Final LayerNorm on CLS token before the last aux_head (which acts as main head)
        # This is not standard Swin but aligns with ViT practice for CLS token.
        self.norm_final_cls = norm_layer(current_dim)

        self.apply(self._init_weights)
        print(f"VisionTransformerSwinHSI: {self.num_stages} stages built. Total Swin blocks: {sum(depths)}. Total Aux Heads: {len(self.all_aux_heads_flat)}")


    def _init_weights(self, m):
        # ... (same as your Step 1 _init_weights)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features(self, x_hsi): # Separated feature extraction
        img_tokens, H, W = self.patch_embed(x_hsi)
        
        current_x = img_tokens
        cls_tokens = self.cls_token.expand(img_tokens.shape[0], -1, -1)
        current_x = torch.cat((cls_tokens, img_tokens), dim=1)
        
        if self.ape:
            current_x = current_x + self.absolute_pos_embed[:, :current_x.size(1)]
        current_x = self.pos_drop(current_x)

        list_all_aux_logits = []
        list_all_hint_features = []
        
        current_H_grid, current_W_grid = H, W # Grid dimensions for image tokens

        for stage_idx, stage_layer in enumerate(self.stages):
            # Pass H, W of the token grid *before* this stage's blocks/merger
            current_x, H_after_stage, W_after_stage, \
            aux_logits_this_stage, hint_features_this_stage = stage_layer(current_x, current_H_grid, current_W_grid)
            
            list_all_aux_logits.extend(aux_logits_this_stage)
            list_all_hint_features.extend(hint_features_this_stage)
            
            current_H_grid, current_W_grid = H_after_stage, W_after_stage
            
        # The last aux_head (from the last block of the last stage) acts as the primary classifier.
        # Apply final norm to CLS token if needed (though aux_heads already process CLS output)
        # final_cls_token_output = self.norm_final_cls(current_x[:, 0]) # current_x is output of last stage
        # The aux_logits_list already contains the output from the last aux_head.

        return list_all_aux_logits, list_all_hint_features

    def forward(self, x_hsi):
        aux_logits, hint_features = self.forward_features(x_hsi)
        return aux_logits, hint_features


if __name__ == '__main__':
    from config import ( # Ensure these are correctly set in config.py for the test
        VIT_IMG_SIZE, VIT_SUB_PATCH_SIZE, NUM_PCA_COMPONENTS,
        SWIN_INITIAL_PATCH_EMBED_DIM, SWIN_DEPTHS, SWIN_NUM_HEADS, SWIN_WINDOW_SIZES,
        SWIN_MLP_RATIO, SWIN_QKV_BIAS, # SWIN_QK_SCALE (can be None)
        SWIN_DROP_RATE, SWIN_ATTN_DROP_RATE, SWIN_DROP_PATH_RATE,
        SWIN_APE, SWIN_PATCH_NORM, BATCH_SIZE
    )
    import config as global_cfg_main

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if global_cfg_main.NUM_CLASSES_ACTUAL is None:
        print("Setting NUM_CLASSES_ACTUAL to 16 for Swin model_swin.py test.")
        global_cfg_main.NUM_CLASSES_ACTUAL = 16
    
    print(f"--- Testing VisionTransformerSwinHSI with CustomBasicLayer ---")
    # CRITICAL: Configure SWIN_DEPTHS, _NUM_HEADS, _WINDOW_SIZES in config.py
    # Example for a 3x3 initial grid (PATCH_SIZE=9, VIT_SUB_PATCH_SIZE=3): Only one stage possible for standard merge.
    # If you want to test multiple stages, ensure PATCH_SIZE/VIT_SUB_PATCH_SIZE create an even grid.
    # For testing, let's assume config has a single stage if initial grid is 3x3.
    # E.g. in config.py: SWIN_DEPTHS = [2], SWIN_NUM_HEADS = [3], SWIN_WINDOW_SIZES = [3] if initial_embed_dim=96
    
    _patch_size = global_cfg_main.PATCH_SIZE
    _sub_patch_size = global_cfg_main.VIT_SUB_PATCH_SIZE
    _initial_grid_h = _patch_size // _sub_patch_size
    _initial_grid_w = _patch_size // _sub_patch_size
    print(f"Config: PATCH_SIZE={_patch_size}, VIT_SUB_PATCH_SIZE={_sub_patch_size}")
    print(f"This implies initial token grid H={_initial_grid_h}, W={_initial_grid_w}")
    print(f"Configured SWIN_DEPTHS: {SWIN_DEPTHS} (num_stages = {len(SWIN_DEPTHS)})")
    
    # Check compatibility of window sizes with initial grid for the first stage
    if SWIN_WINDOW_SIZES[0] > _initial_grid_h or SWIN_WINDOW_SIZES[0] > _initial_grid_w :
        print(f"ERROR: SWIN_WINDOW_SIZES[0]={SWIN_WINDOW_SIZES[0]} is larger than initial token grid ({_initial_grid_h}x{_initial_grid_w}). Adjust config.")
        exit()
    
    # Check if multiple stages are configured with an odd initial grid (problematic for merging)
    if len(SWIN_DEPTHS) > 1 and (_initial_grid_h % 2 != 0 or _initial_grid_w % 2 != 0):
        print(f"WARNING: Multiple Swin stages ({len(SWIN_DEPTHS)}) configured, but initial token grid ({_initial_grid_h}x{_initial_grid_w}) "
              "is odd. Standard PatchMerging expects even dimensions and will truncate. "
              "Consider adjusting PATCH_SIZE/VIT_SUB_PATCH_SIZE for an even grid or expect truncation.")


    model_swin_test = VisionTransformerSwinHSI(
        img_size_hw=VIT_IMG_SIZE, sub_patch_size=VIT_SUB_PATCH_SIZE,
        in_chans=NUM_PCA_COMPONENTS, num_classes=global_cfg_main.NUM_CLASSES_ACTUAL,
        initial_embed_dim=SWIN_INITIAL_PATCH_EMBED_DIM,
        depths=SWIN_DEPTHS, num_heads_list=SWIN_NUM_HEADS, window_sizes=SWIN_WINDOW_SIZES,
        mlp_ratio=SWIN_MLP_RATIO, qkv_bias=SWIN_QKV_BIAS, qk_scale=SWIN_QK_SCALE,
        drop_rate=SWIN_DROP_RATE, attn_drop_rate=SWIN_ATTN_DROP_RATE, drop_path_rate=SWIN_DROP_PATH_RATE,
        ape=SWIN_APE, patch_norm=SWIN_PATCH_NORM,
        use_cls_token=True # Must be True for this self-distillation design
    ).to(device)

    print(f"\nModel Instantiated: {type(model_swin_test)}")

    dummy_input = torch.randn(BATCH_SIZE//2 if BATCH_SIZE > 1 else 1, # Smaller batch for faster test
                              NUM_PCA_COMPONENTS, VIT_IMG_SIZE[0], VIT_IMG_SIZE[1]).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    try:
        logits_list, features_list = model_swin_test(dummy_input)
        print(f"\n--- Model Forward Output (Test) ---")
        print(f"Output logits_list length: {len(logits_list)} (should be == total Swin blocks = {sum(SWIN_DEPTHS)})")
        if logits_list:
            print(f"  Shape of first logits: {logits_list[0].shape}")
            print(f"  Shape of last logits: {logits_list[-1].shape}")
        else: print("Output logits_list is empty.")
        
        print(f"Output features_list length: {len(features_list)} (should be == total Swin blocks)")
        if features_list:
            print(f"  Shape of first hint features: {features_list[0].shape}")
            print(f"  Shape of last hint features: {features_list[-1].shape}")
        else: print("Output features_list is empty.")
        print("Full forward pass with CustomBasicLayer executed.")

        # Optional: torchinfo summary
        from torchinfo import summary
        summary(model_swin_test, input_data=dummy_input, verbose=0) # verbose=0 for concise
        print("\nTorchinfo summary generated.")

    except Exception as e:
        print(f"Error during model_swin_test forward pass: {e}")
        import traceback
        traceback.print_exc()