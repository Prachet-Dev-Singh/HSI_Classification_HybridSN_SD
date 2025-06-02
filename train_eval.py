# train_eval.py
import torch
import numpy as np
from tqdm import tqdm

from losses import compute_total_loss # Keep for train_one_epoch
import config # To get NUM_TRANSFORMER_BLOCKS (or equivalent total Swin blocks) for evaluation

def train_one_epoch(model, dataloader, optimizer, device, current_epoch, total_epochs):
    model.train()
    total_loss_epoch = 0.0
    
    epoch_loss_components = {
        "ce_final": 0.0, "ce_aux_sum": 0.0,
        "kl_distill_sum": 0.0, "l2_hint_sum": 0.0
    }

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch}/{total_epochs} [Training]", leave=False)
    for patches, labels in progress_bar:
        patches, labels = patches.to(device), labels.to(device)
        
        optimizer.zero_grad()
        model_outputs = model(patches) 
        loss, batch_loss_components = compute_total_loss(model_outputs, labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected at epoch {current_epoch}, batch. Loss: {loss.item()}. Components: {batch_loss_components}")
            continue 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_epoch += loss.item()
        for key in epoch_loss_components:
            if key in batch_loss_components:
                 epoch_loss_components[key] += batch_loss_components[key]
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0, {}
    avg_epoch_loss = total_loss_epoch / num_batches
    avg_epoch_components = {k: v / num_batches for k, v in epoch_loss_components.items()}
    return avg_epoch_loss, avg_epoch_components

def evaluate(model, dataloader, device, current_epoch=None, total_epochs=None, eval_type="Validation", evaluate_all_heads=False): # MODIFIED HERE
    """
    Evaluates the model. Can evaluate all heads or just the final head.
    ... (docstring as provided before) ...
    """
    model.eval()
    
    # Determine number of heads to evaluate based on the model's structure
    # For Swin, it's sum(config.SWIN_DEPTHS). For old ViT, config.NUM_TRANSFORMER_BLOCKS
    # This assumes model_outputs[0] is the list of logits from all heads/blocks
    # A more robust way might be to get num_heads from model.all_aux_heads_flat if available
    # For now, let's assume the first forward pass gives us the number of logit tensors
    with torch.no_grad():
        sample_patches, _ = next(iter(dataloader))
        sample_patches = sample_patches.to(device)
        num_actual_heads_in_model = len(model(sample_patches)[0]) # model_outputs[0] is logits_list

    num_heads_to_evaluate_from_model = num_actual_heads_in_model
    
    num_heads_to_process = num_heads_to_evaluate_from_model if evaluate_all_heads else 1 # If not all_heads, just process last one conceptually
    
    if evaluate_all_heads:
        total_correct_per_head = [0] * num_heads_to_process
        all_preds_per_head_collect = [[] for _ in range(num_heads_to_process)]
    else: # For final head only
        total_correct_final_head = 0
        all_preds_final_head_collect = []

    total_samples = 0
    all_labels_collect = []

    desc_text = f"Epoch {current_epoch}/{total_epochs} [{eval_type}]" if current_epoch else f"[{eval_type}]"
    if evaluate_all_heads: desc_text += " (All Heads)"
    progress_bar = tqdm(dataloader, desc=desc_text, leave=False)

    with torch.no_grad():
        for patches, labels_batch in progress_bar:
            patches, labels_batch = patches.to(device), labels_batch.to(device)
            
            model_outputs = model(patches)
            all_logits_list = model_outputs[0] # List of logits from all aux heads

            total_samples += labels_batch.size(0)
            all_labels_collect.extend(labels_batch.cpu().numpy())

            if evaluate_all_heads:
                assert len(all_logits_list) == num_heads_to_process, \
                    f"Mismatch: Model produced {len(all_logits_list)} logit sets, expected {num_heads_to_process} for all_heads eval."
                for head_idx in range(num_heads_to_process):
                    logits_current_head = all_logits_list[head_idx]
                    _, predicted_current_head = torch.max(logits_current_head.data, 1)
                    total_correct_per_head[head_idx] += (predicted_current_head == labels_batch).sum().item()
                    all_preds_per_head_collect[head_idx].extend(predicted_current_head.cpu().numpy())
            else: # Only evaluate final head
                final_logits = all_logits_list[-1] # Always take the last one as the "final" head
                _, predicted_final_head = torch.max(final_logits.data, 1)
                total_correct_final_head += (predicted_final_head == labels_batch).sum().item()
                all_preds_final_head_collect.extend(predicted_final_head.cpu().numpy())
                
    if total_samples == 0:
        if evaluate_all_heads:
            return [0.0] * num_heads_to_process, [np.array([])] * num_heads_to_process, np.array([])
        else:
            return 0.0, np.array([]), np.array([])

    true_labels_np = np.array(all_labels_collect)

    if evaluate_all_heads:
        accuracies = [(100.0 * correct / total_samples) for correct in total_correct_per_head]
        all_preds_list_np = [np.array(preds) for preds in all_preds_per_head_collect]
        return accuracies, all_preds_list_np, true_labels_np
    else:
        accuracy_final_head = 100.0 * total_correct_final_head / total_samples
        all_preds_final_head_np = np.array(all_preds_final_head_collect)
        return accuracy_final_head, all_preds_final_head_np, true_labels_np


if __name__ == '__main__':
    # --- Testing train_eval.py functions (basic structural test) ---
    print("--- Testing train_eval.py functions (basic structural test) ---")

    # Import config and utils for testing
    import config as cfg_test # Use a different alias to avoid confusion if main.py also imports config
    from utils import set_seeds
    # For testing, we need a model. Let's use a simplified version of VisionTransformerSwinHSI shell.
    # Or, if your old model.py is still around for testing the original ViT path:
    # from model import VisionTransformerWithAuxHeads # For testing original ViT evaluate path
    
    # --- Test for evaluate_all_heads path ---
    # We need a model that outputs a list of logits.
    # Let's use VisionTransformerSwinHSI from model_swin.py for this test
    try:
        from model_swin import VisionTransformerSwinHSI # Assuming this is your new model
        swin_model_available = True
    except ImportError:
        print("Warning: model_swin.py or VisionTransformerSwinHSI not found for detailed test.")
        swin_model_available = False

    from hsi_dataset import HSIDataset 
    from torch.utils.data import DataLoader
    import torch.optim as optim

    set_seeds(cfg_test.RANDOM_SEED)
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for test: {device_test}")

    if cfg_test.NUM_CLASSES_ACTUAL is None:
        print("Setting cfg_test.NUM_CLASSES_ACTUAL to 3 for test.")
        cfg_test.NUM_CLASSES_ACTUAL = 3
    
    # Dummy data for testing evaluate
    dummy_patches_eval = np.random.randn(cfg_test.BATCH_SIZE, cfg_test.NUM_PCA_COMPONENTS, cfg_test.PATCH_SIZE, cfg_test.PATCH_SIZE).astype(np.float32)
    dummy_labels_eval = np.random.randint(0, cfg_test.NUM_CLASSES_ACTUAL, cfg_test.BATCH_SIZE)
    dummy_dataset_eval = HSIDataset(dummy_patches_eval, dummy_labels_eval) # HSIDataset expects (N,H,W,C)
    # Correction for HSIDataset input if dummy_patches_eval is already (N,C,H,W)
    # If HSIDataset permutes, ensure dummy_patches are (N,H,W,C)
    # My dummy_patches_eval is (N,C,H,W) if NUM_PCA_COMPONENTS is C. HSIDataset expects (N,H,W,C_pca)
    # Let's fix dummy_patches_eval creation to match HSIDataset expectation
    dummy_patches_eval_correct_format = np.random.randn(
        cfg_test.BATCH_SIZE, cfg_test.PATCH_SIZE, cfg_test.PATCH_SIZE, cfg_test.NUM_PCA_COMPONENTS
    ).astype(np.float32)
    dummy_dataset_eval = HSIDataset(dummy_patches_eval_correct_format, dummy_labels_eval)
    dummy_dataloader_eval = DataLoader(dummy_dataset_eval, batch_size=cfg_test.BATCH_SIZE)


    if swin_model_available:
        print("\n--- Testing evaluate with evaluate_all_heads=True using VisionTransformerSwinHSI ---")
        # Configure a simple Swin model for test
        test_model_swin = VisionTransformerSwinHSI(
            img_size_hw=(cfg_test.PATCH_SIZE, cfg_test.PATCH_SIZE),
            sub_patch_size=cfg_test.VIT_SUB_PATCH_SIZE,
            in_chans=cfg_test.NUM_PCA_COMPONENTS,
            num_classes=cfg_test.NUM_CLASSES_ACTUAL,
            initial_embed_dim=cfg_test.SWIN_INITIAL_PATCH_EMBED_DIM, # Ensure this is in config
            depths=cfg_test.SWIN_DEPTHS if hasattr(cfg_test, 'SWIN_DEPTHS') else [2], # Use config or default
            num_heads_list=cfg_test.SWIN_NUM_HEADS if hasattr(cfg_test, 'SWIN_NUM_HEADS') else [2],
            window_sizes=cfg_test.SWIN_WINDOW_SIZES if hasattr(cfg_test, 'SWIN_WINDOW_SIZES') else [cfg_test.PATCH_SIZE // cfg_test.VIT_SUB_PATCH_SIZE],
            mlp_ratio=cfg_test.SWIN_MLP_RATIO if hasattr(cfg_test, 'SWIN_MLP_RATIO') else 2.0, # Simpler MLP
            qkv_bias=cfg_test.SWIN_QKV_BIAS if hasattr(cfg_test, 'SWIN_QKV_BIAS') else True,
            drop_rate=cfg_test.SWIN_DROP_RATE if hasattr(cfg_test, 'SWIN_DROP_RATE') else 0.0,
            attn_drop_rate=cfg_test.SWIN_ATTN_DROP_RATE if hasattr(cfg_test, 'SWIN_ATTN_DROP_RATE') else 0.0,
            drop_path_rate=cfg_test.SWIN_DROP_PATH_RATE if hasattr(cfg_test, 'SWIN_DROP_PATH_RATE') else 0.0,
            ape=cfg_test.SWIN_APE if hasattr(cfg_test, 'SWIN_APE') else False,
            patch_norm=cfg_test.SWIN_PATCH_NORM if hasattr(cfg_test, 'SWIN_PATCH_NORM') else True,
            use_cls_token=True
        ).to(device_test)

        accuracies_all, preds_all, labels_all = evaluate(
            test_model_swin, dummy_dataloader_eval, device_test, 
            eval_type="Test All Heads", evaluate_all_heads=True
        )
        print(f"  Accuracies per head: {accuracies_all}")
        if preds_all and isinstance(preds_all, list) and len(preds_all) > 0:
            for i, preds_h in enumerate(preds_all):
                print(f"  Predictions from head {i+1} shape: {preds_h.shape if isinstance(preds_h, np.ndarray) else 'N/A'}")
        print(f"  True labels shape: {labels_all.shape if isinstance(labels_all, np.ndarray) else 'N/A'}")

    else: # Fallback to testing with original ViT if Swin model is not ready/imported
        print("\n--- Testing evaluate (final head only) using original VisionTransformerWithAuxHeads ---")
        # This part of the test requires original ViT params in config (EMBED_DIM etc.)
        if hasattr(cfg_test, 'EMBED_DIM'): # Check if old ViT params exist
            test_model_orig_vit = VisionTransformerWithAuxHeads(
                img_size=cfg_test.VIT_IMG_SIZE, vit_sub_patch_size=cfg_test.VIT_SUB_PATCH_SIZE,
                in_channels=cfg_test.NUM_PCA_COMPONENTS, num_classes=cfg_test.NUM_CLASSES_ACTUAL,
                embed_dim=cfg_test.EMBED_DIM, num_heads=cfg_test.NUM_HEADS,
                num_transformer_blocks= getattr(cfg_test, 'NUM_TRANSFORMER_BLOCKS', 2), # Default to 2 blocks for test
                mlp_hidden_layers_transformer=getattr(cfg_test, 'MLP_TRANSFORMER_HIDDEN_LAYERS', [64,64]), 
                dropout_rate=getattr(cfg_test, 'DROPOUT_RATE', 0.1)
            ).to(device_test)
            accuracy_final, _, _ = evaluate(test_model_orig_vit, dummy_dataloader_eval, device_test, eval_type="Test Final Head", evaluate_all_heads=False)
            print(f"  Accuracy (final head only): {accuracy_final:.2f}%")
        else:
            print("Skipping original ViT evaluate test as EMBED_DIM not found in config (likely Swin config is active).")


    print("\ntrain_eval.py basic test finished.")