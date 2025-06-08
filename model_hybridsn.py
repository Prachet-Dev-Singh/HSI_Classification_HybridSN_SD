# model_hybridsn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_

import config # Import config to access parameters

# --- Activation Layer Helper ---
def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True # Expects (B, SeqLen, EmbedDim)
        )
        # No MLP here, as the MLPAuxHead will follow.
        # We could add one if desired: self.mlp = ...

    def forward(self, x):
        # x is expected to be (B, embed_dim) - the pooled features
        B, E = x.shape
        assert E == self.embed_dim, "Input feature dimension mismatch"

        # Reshape to (B, 1, embed_dim) to act as a sequence of length 1
        x_seq = x.unsqueeze(1) # (B, 1, E)
        
        normed_x = self.norm(x_seq)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x) # Query, Key, Value are the same
        
        # Residual connection (important for attention layers)
        # Since input to attention was normed_x, residual should be from x_seq or x_normed
        # Let's add residual from the input to the norm (x_seq)
        attended_features_seq = x_seq + attn_output # (B, 1, E)
        
        # Squeeze back to (B, embed_dim) for the MLPAuxHead
        attended_features = attended_features_seq.squeeze(1) # (B, E)
        
        return attended_features


# --- NEW MLP Auxiliary Head ---
class MLPAuxHead(nn.Module):
    def __init__(self, in_features, num_classes, mlp_hidden_layers_dims, dropout_rate, activation_fn_str="relu"):
        """
        Args:
            in_features (int): Number of input features (e.g., channels after pooling).
            num_classes (int): Number of output classes.
            mlp_hidden_layers_dims (list of int): List of hidden layer dimensions.
                                                e.g., [128, 128, 128, 128] for a 5-layer MLP.
                                                Input -> H1 -> H2 -> H3 -> H4 -> Output.
            dropout_rate (float): Dropout rate.
            activation_fn_str (str): Activation function string ("relu" or "gelu").
        """
        super().__init__()
        
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"MLPAuxHead: num_classes must be a positive integer, got {num_classes}")
        # Allow empty mlp_hidden_layers_dims to mean a single linear layer (like original AuxHead)
        # if not mlp_hidden_layers_dims: 
        #      raise ValueError("MLPAuxHead: mlp_hidden_layers_dims cannot be empty for a multi-layer MLP head.")

        layers = []
        current_dim = in_features
        activation_fn = get_activation(activation_fn_str)

        # Hidden layers
        for h_dim in mlp_hidden_layers_dims: # If list is empty, this loop is skipped
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        # No dropout usually after the very final logit layer in a classifier head
        
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, x): # x is (B, in_features)
        return self.mlp_head(x)

# --- Main HybridSN Model with Self-Attention for Student Aux Heads ---
class HybridSNwithDistillation(nn.Module):
    def __init__(self, input_bands, patch_size_spatial, num_classes):
        super().__init__()
        self.input_bands = input_bands
        self.patch_size_spatial = patch_size_spatial
        self.num_classes = num_classes

        # MLP AuxHead config
        self.aux_head_mlp_dims = getattr(config, 'HYBRIDSN_AUX_HEAD_MLP_DIMS', [128, 128, 128, 128])
        self.aux_head_mlp_dropout = getattr(config, 'HYBRIDSN_AUX_HEAD_MLP_DROPOUT', 0.2)
        self.aux_head_activation = getattr(config, 'HYBRIDSN_AUX_HEAD_ACTIVATION', config.HYBRIDSN_ACTIVATION)

        # Student Aux Self-Attention config
        self.student_aux_use_sa = getattr(config, 'STUDENT_AUX_USE_SELF_ATTENTION', False)
        self.student_aux_sa_num_heads = getattr(config, 'STUDENT_AUX_SA_NUM_HEADS', 4)
        self.student_aux_sa_dropout = getattr(config, 'STUDENT_AUX_SA_DROPOUT', 0.1)

        # --- ModuleLists for layers and heads ---
        self.conv3d_blocks = nn.ModuleList()
        self.sa_after_3d_conv = nn.ModuleList() # SA layers for 3D conv outputs
        self.aux_heads_after_3d_conv = nn.ModuleList()
        self.hint_projectors_3d = nn.ModuleList()

        # For 2D Conv
        # self.conv2d_module will be nn.Sequential
        self.sa_after_2d_conv = None # Single SA layer for 2D conv output
        self.aux_head_after_2d_conv = None
        self.hint_projector_2d = None

        self.fc_blocks = nn.ModuleList()
        # Aux heads for FC hidden layers were removed as per your previous request.
        # If you re-add them, you'd also add corresponding SA layers here.

        # Calculate spatial dimensions dynamically (as before)
        _h_curr, _w_curr, _d_curr, _c_curr = patch_size_spatial, patch_size_spatial, input_bands, 1
        for i, layer_conf_3d in enumerate(config.HYBRIDSN_CONV3D_LAYERS):
            k_d,k_h_3d,k_w_3d=layer_conf_3d["kernel_size"]; p_d,p_h_3d,p_w_3d=layer_conf_3d["padding"]
            actual_k_d=min(k_d, _d_curr if i>0 else input_bands)
            _d_curr=(_d_curr if i>0 else input_bands)+2*p_d-actual_k_d+1
            _h_curr=(_h_curr+2*p_h_3d-k_h_3d)+1; _w_curr=(_w_curr+2*p_w_3d-k_w_3d)+1
            _c_curr=layer_conf_3d["out_channels"]
        _h_after_3d_convs, _w_after_3d_convs = _h_curr, _w_curr
        
        _h_after_2d_conv = (_h_after_3d_convs + 2*config.HYBRIDSN_CONV2D_PADDING - config.HYBRIDSN_CONV2D_KERNEL_SIZE)+1
        _w_after_2d_conv = (_w_after_3d_convs + 2*config.HYBRIDSN_CONV2D_PADDING - config.HYBRIDSN_CONV2D_KERNEL_SIZE)+1
        
        fc_input_features_for_teacher_hint_calc = config.HYBRIDSN_CONV2D_OUT_CHANNELS * _h_after_2d_conv * _w_after_2d_conv
        self.teacher_hint_dim = config.HYBRIDSN_FC_HIDDEN_UNITS[-1] if config.HYBRIDSN_FC_HIDDEN_UNITS else \
                                fc_input_features_for_teacher_hint_calc
        print(f"L2 Hint Loss Teacher Feature Dimension will be: {self.teacher_hint_dim}")

        # --- 1. 3D Convolutional Layers ---
        print("--- Building HybridSN: 3D Conv Layers ---")
        current_in_channels_for_3d_conv, current_spectral_dim = 1, self.input_bands
        current_spatial_dim_h, current_spatial_dim_w = self.patch_size_spatial, self.patch_size_spatial
        for i, layer_conf in enumerate(config.HYBRIDSN_CONV3D_LAYERS):
            out_c = layer_conf["out_channels"]
            k_d,k_h,k_w=layer_conf["kernel_size"]; p_d,p_h,p_w=layer_conf["padding"]
            actual_k_d=min(k_d,current_spectral_dim)
            next_spectral_dim=(current_spectral_dim+2*p_d-actual_k_d)+1
            next_spatial_dim_h=(current_spatial_dim_h+2*p_h-k_h)+1
            next_spatial_dim_w=(current_spatial_dim_w+2*p_w-k_w)+1
            print(f"  Layer {i+1} (3D Conv): OutCh={out_c}, OutDims (D,H,W)=({next_spectral_dim},{next_spatial_dim_h},{next_spatial_dim_w})")
            conv3d=nn.Conv3d(current_in_channels_for_3d_conv,out_c,(actual_k_d,k_h,k_w),padding=(p_d,p_h,p_w))
            block_layers=[conv3d]
            if config.HYBRIDSN_USE_BATCHNORM3D: block_layers.append(nn.BatchNorm3d(out_c))
            block_layers.append(get_activation(config.HYBRIDSN_ACTIVATION))
            self.conv3d_blocks.append(nn.Sequential(*block_layers))
            
            if self.student_aux_use_sa:
                self.sa_after_3d_conv.append(FeatureSelfAttention(out_c, self.student_aux_sa_num_heads, self.student_aux_sa_dropout))
            self.aux_heads_after_3d_conv.append(MLPAuxHead(out_c,self.num_classes,self.aux_head_mlp_dims,self.aux_head_mlp_dropout,self.aux_head_activation))
            if out_c!=self.teacher_hint_dim: self.hint_projectors_3d.append(nn.Linear(out_c,self.teacher_hint_dim))
            else: self.hint_projectors_3d.append(nn.Identity())
            current_in_channels_for_3d_conv,current_spectral_dim=out_c,next_spectral_dim
            current_spatial_dim_h,current_spatial_dim_w=next_spatial_dim_h,next_spatial_dim_w
        
        self.final_3d_out_channels,self.final_spectral_dim_after_3d=current_in_channels_for_3d_conv,current_spectral_dim
        self.final_spatial_h_after_3d,self.final_spatial_w_after_3d=current_spatial_dim_h,current_spatial_dim_w

        # --- 2. Reshape and 2D Convolutional Layer ---
        print(f"--- Building HybridSN: Reshape and 2D Conv Layer ---")
        in_chans_for_2d_conv = self.final_3d_out_channels * self.final_spectral_dim_after_3d
        conv2d_layers_list = [nn.Conv2d(in_chans_for_2d_conv, config.HYBRIDSN_CONV2D_OUT_CHANNELS, config.HYBRIDSN_CONV2D_KERNEL_SIZE, padding=config.HYBRIDSN_CONV2D_PADDING)]
        self.final_spatial_h_after_2d, self.final_spatial_w_after_2d = _h_after_2d_conv, _w_after_2d_conv
        print(f"  2D Conv Layer: OutCh={config.HYBRIDSN_CONV2D_OUT_CHANNELS}, OutSpatial ({self.final_spatial_h_after_2d}x{self.final_spatial_w_after_2d})")
        if config.HYBRIDSN_USE_BATCHNORM2D: conv2d_layers_list.append(nn.BatchNorm2d(config.HYBRIDSN_CONV2D_OUT_CHANNELS))
        conv2d_layers_list.append(get_activation(config.HYBRIDSN_ACTIVATION))
        self.conv2d_module = nn.Sequential(*conv2d_layers_list)
        
        if self.student_aux_use_sa:
            self.sa_after_2d_conv = FeatureSelfAttention(config.HYBRIDSN_CONV2D_OUT_CHANNELS, self.student_aux_sa_num_heads, self.student_aux_sa_dropout)
        self.aux_head_after_2d_conv = MLPAuxHead(config.HYBRIDSN_CONV2D_OUT_CHANNELS, self.num_classes, self.aux_head_mlp_dims, self.aux_head_mlp_dropout, self.aux_head_activation)
        if config.HYBRIDSN_CONV2D_OUT_CHANNELS != self.teacher_hint_dim: self.hint_projector_2d = nn.Linear(config.HYBRIDSN_CONV2D_OUT_CHANNELS, self.teacher_hint_dim)
        else: self.hint_projector_2d = nn.Identity()

        # --- 3. Fully Connected Layers (No Aux Heads/SA for hidden FCs) ---
        print(f"--- Building HybridSN: FC Layers ---")
        fc_input_dim = config.HYBRIDSN_CONV2D_OUT_CHANNELS * self.final_spatial_h_after_2d * self.final_spatial_w_after_2d
        self.fc_blocks = nn.ModuleList()
        current_fc_in_features = fc_input_dim
        for i, hidden_units in enumerate(config.HYBRIDSN_FC_HIDDEN_UNITS):
            fc_layer = nn.Linear(current_fc_in_features, hidden_units)
            block_layers = [nn.Dropout(config.HYBRIDSN_DROPOUT_RATE_FC), fc_layer, get_activation(config.HYBRIDSN_ACTIVATION)]
            self.fc_blocks.append(nn.Sequential(*block_layers))
            print(f"  FC Hidden Layer {i+1}: InFeatures={current_fc_in_features}, OutFeatures={hidden_units}")
            current_fc_in_features = hidden_units
        
        self.output_fc_layer = nn.Linear(current_fc_in_features, num_classes)
        print(f"  Output FC Layer: InFeatures={current_fc_in_features}, OutFeatures={num_classes}")

        self.apply(self._init_weights)
        print("HybridSNwithDistillation model (MLPAuxHeads for CNNs with optional SA for students) built successfully.")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02);
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            
    def _process_student_head(self, x_feat, sa_module_if_any, aux_head_module, hint_projector_module, 
                              all_logits_list, all_hints_list, pool_type="2d_spatial"):
        pooled_features = None # Calculate pooled_features based on pool_type and x_feat.ndim
        if pool_type == "3d_all" and x_feat.ndim == 5:
            pooled_features = F.adaptive_avg_pool3d(x_feat, (1, 1, 1)).flatten(start_dim=1)
        elif pool_type == "2d_spatial" and x_feat.ndim == 4:
            pooled_features = F.adaptive_avg_pool2d(x_feat, (1, 1)).flatten(start_dim=1)
        # Removed the x_feat.ndim == 2 case from here as FC student heads are removed
        else:
            # Fallback pooling
            num_channels_fallback = x_feat.shape[1] if x_feat.ndim > 1 else 1
            if x_feat.numel() == 0 or x_feat.shape[0] == 0 or (x_feat.ndim > 1 and x_feat.numel() // x_feat.shape[0] == 0):
                 pooled_features = torch.zeros(x_feat.shape[0], num_channels_fallback, device=x_feat.device)
                 print(f"Warning: Zero-element features encountered before pooling for aux head. Shape: {x_feat.shape}")
            else:
                pooled_features = x_feat.view(x_feat.shape[0], num_channels_fallback, -1).mean(dim=2)
                if x_feat.ndim != 5 and x_feat.ndim != 4 : # Only print warning if not explicitly handled
                    print(f"Warning: Unexpected feature map for aux head. Shape: {x_feat.shape}. Pooled to: {pooled_features.shape}")


        features_for_mlp_aux_head = pooled_features
        if self.student_aux_use_sa and sa_module_if_any is not None:
            features_for_mlp_aux_head = sa_module_if_any(pooled_features) # Apply SA

        logits = aux_head_module(features_for_mlp_aux_head) # MLPAuxHead
        
        # Hint features are taken from *before* this student-specific SA layer, after pooling
        projected_hint = hint_projector_module(pooled_features) 

        all_logits_list.append(logits)
        all_hints_list.append(projected_hint.detach())

    def forward(self, x_hsi_patch):
        x = x_hsi_patch.unsqueeze(1)
        all_logits, all_hint_features = [], []

        # 1. 3D Convolutional Layers
        for i, conv3d_block_module in enumerate(self.conv3d_blocks):
            x = conv3d_block_module(x)
            sa_module = self.sa_after_3d_conv[i] if self.student_aux_use_sa else None
            self._process_student_head(
                x, sa_module, self.aux_heads_after_3d_conv[i], self.hint_projectors_3d[i],
                all_logits, all_hint_features, pool_type="3d_all"
            )
        
        B, C3d, D3d, H3d, W3d = x.shape; x = x.reshape(B, C3d * D3d, H3d, W3d)
        
        # 2. 2D Convolutional Layer
        x = self.conv2d_module(x)
        sa_module_2d = self.sa_after_2d_conv if self.student_aux_use_sa else None
        self._process_student_head(
            x, sa_module_2d, self.aux_head_after_2d_conv, self.hint_projector_2d,
            all_logits, all_hint_features, pool_type="2d_spatial"
        )

        x = x.reshape(B, -1)

        # 3. Hidden Fully Connected Layers (pass through, no aux heads)
        for fc_block_module in self.fc_blocks:
            x = fc_block_module(x) 
        
        # 4. Final Output Layer (Main Head - no student SA here)
        main_output_logits = self.output_fc_layer(x)
        all_logits.append(main_output_logits)
        
        final_hint = x # Features before the last linear layer
        if final_hint.shape[1] != self.teacher_hint_dim:
             # This should ideally not happen if teacher_hint_dim is correctly set based on last hidden FC or 2D conv output
             print(f"CRITICAL WARNING: Final hint (teacher) dimension {final_hint.shape[1]} "
                   f"does NOT match configured teacher_hint_dim {self.teacher_hint_dim}. "
                   "L2 Loss will be incorrect. This usually means teacher_hint_dim "
                   "was not correctly derived from the output of the last hidden FC layer "
                   "or the (pooled) 2D conv output if no hidden FCs.")
             # Attempting a non-learnable resize for now to prevent crash, but this is a bug indicator.
             # A learnable projection is not appropriate for the teacher hint itself.
             if final_hint.shape[1] > self.teacher_hint_dim :
                 final_hint = final_hint[:, :self.teacher_hint_dim] # Truncate
             else: # Pad (less ideal) - this scenario needs fixing in teacher_hint_dim calculation
                 padding_needed = self.teacher_hint_dim - final_hint.shape[1]
                 final_hint = F.pad(final_hint, (0, padding_needed))

        all_hint_features.append(final_hint.detach()) 

        return all_logits, all_hint_features


if __name__ == '__main__':
    from config import PATCH_SIZE, NUM_PCA_COMPONENTS, BATCH_SIZE
    import config as global_cfg_main_test # Alias to access and potentially set config values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Ensure necessary config values are present for the test ---
    if global_cfg_main_test.NUM_CLASSES_ACTUAL is None:
        print("Setting NUM_CLASSES_ACTUAL to 16 for HybridSN model_hybridsn.py test.")
        global_cfg_main_test.NUM_CLASSES_ACTUAL = 16
    
    # Ensure MLP AuxHead config params are available
    if not hasattr(global_cfg_main_test, 'HYBRIDSN_AUX_HEAD_MLP_DIMS'):
        print("Setting default HYBRIDSN_AUX_HEAD_MLP_DIMS for test.")
        global_cfg_main_test.HYBRIDSN_AUX_HEAD_MLP_DIMS = [128,128,128,128] # Default for MLPAuxHead
    if not hasattr(global_cfg_main_test, 'HYBRIDSN_AUX_HEAD_MLP_DROPOUT'):
        global_cfg_main_test.HYBRIDSN_AUX_HEAD_MLP_DROPOUT = 0.2
    if not hasattr(global_cfg_main_test, 'HYBRIDSN_AUX_HEAD_ACTIVATION'):
        global_cfg_main_test.HYBRIDSN_AUX_HEAD_ACTIVATION = "relu"

    # Ensure Student Aux SA config params are available (CRITICAL FOR THIS TEST)
    if not hasattr(global_cfg_main_test, 'STUDENT_AUX_USE_SELF_ATTENTION'):
        print("Setting default STUDENT_AUX_USE_SELF_ATTENTION = True for test.")
        global_cfg_main_test.STUDENT_AUX_USE_SELF_ATTENTION = True # Enable SA for this test
    if not hasattr(global_cfg_main_test, 'STUDENT_AUX_SA_NUM_HEADS'):
        print("Setting default STUDENT_AUX_SA_NUM_HEADS = 4 for test.")
        global_cfg_main_test.STUDENT_AUX_SA_NUM_HEADS = 4
    if not hasattr(global_cfg_main_test, 'STUDENT_AUX_SA_DROPOUT'):
        print("Setting default STUDENT_AUX_SA_DROPOUT = 0.1 for test.")
        global_cfg_main_test.STUDENT_AUX_SA_DROPOUT = 0.1
    # --- End of config setup for test ---

    print(f"--- Testing HybridSNwithDistillation (MLPAuxHeads, Student SA, No FC Aux Heads) ---")
    print(f"Input Config: PCA Bands={NUM_PCA_COMPONENTS}, Patch Size={PATCH_SIZE}x{PATCH_SIZE}")
    print(f"AuxHead MLP Dims: {getattr(global_cfg_main_test, 'HYBRIDSN_AUX_HEAD_MLP_DIMS', 'Not Set')}")
    # Print SA config being used for the test
    print(f"Using Student Aux Self-Attention: {global_cfg_main_test.STUDENT_AUX_USE_SELF_ATTENTION}")
    if global_cfg_main_test.STUDENT_AUX_USE_SELF_ATTENTION:
        print(f"  SA Num Heads: {global_cfg_main_test.STUDENT_AUX_SA_NUM_HEADS}")
        print(f"  SA Dropout: {global_cfg_main_test.STUDENT_AUX_SA_DROPOUT}")

    try:
        model_hybridsn_test = HybridSNwithDistillation(
            input_bands=NUM_PCA_COMPONENTS,
            patch_size_spatial=PATCH_SIZE,
            num_classes=global_cfg_main_test.NUM_CLASSES_ACTUAL
        ).to(device)
        print(f"\nModel Instantiated: {type(model_hybridsn_test)}")

        test_batch = BATCH_SIZE // 2 if BATCH_SIZE > 1 else 1
        if test_batch == 0: test_batch = 1 # Ensure batch size is at least 1 for test
        dummy_input = torch.randn(test_batch, NUM_PCA_COMPONENTS, PATCH_SIZE, PATCH_SIZE).to(device)
        print(f"Dummy input shape for forward pass: {dummy_input.shape}")
        
        logits_list, features_list = model_hybridsn_test(dummy_input)
        
        print(f"\n--- Model Forward Output (Test) ---")
        EXPECTED_HEADS = len(config.HYBRIDSN_CONV3D_LAYERS) + 1 + 1 # 3D_heads + 2D_head + Final_head
        print(f"Expected #Heads = {EXPECTED_HEADS}")
        print(f"Output logits_list length: {len(logits_list)}")
        if logits_list:
            for i, lgs in enumerate(logits_list): print(f"  Logits from head {i+1}: {lgs.shape}")
        
        print(f"Output features_list length: {len(features_list)}")
        if features_list:
            for i, fts in enumerate(features_list): print(f"  Hint features for head {i+1}: {fts.shape}")
        print("Basic forward pass for HybridSN (MLPAuxHeads, Student SA, No FC Aux Heads) executed successfully.")
        
        from torchinfo import summary
        summary(model_hybridsn_test, input_size=(test_batch, NUM_PCA_COMPONENTS, PATCH_SIZE, PATCH_SIZE), verbose=0)
        print("\nTorchinfo summary generated.")

    except Exception as e:
        print(f"Error during HybridSN model_hybridsn.py test: {e}")
        import traceback
        traceback.print_exc()