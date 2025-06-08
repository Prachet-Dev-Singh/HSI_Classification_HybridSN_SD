# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

# Import from our project modules
import config # Ensure this config.py has the HYBRIDSN_* parameters
from utils import set_seeds, ensure_dir
from download_data import download_indian_pines
from data_preprocessing import get_prepared_data
from hsi_dataset import HSIDataset
# MODIFIED: Import the new HybridSN model
# from model_swin import VisionTransformerSwinHSI # Old Swin model
from model_hybridsn import HybridSNwithDistillation # NEW HybridSN model

from train_eval import train_one_epoch, evaluate # evaluate function should handle multiple heads

def main_experiment():
    # --- 1. Initial Setup ---
    set_seeds(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = "./experiment_results"
    ensure_dir(base_results_dir)
    
    # MODIFIED: Experiment name for clarity
    experiment_dir_name = f"{config.DATASET_NAME}_HybridSNDSD_{timestamp}"
    current_experiment_dir = os.path.join(base_results_dir, experiment_dir_name)
    ensure_dir(current_experiment_dir)

    model_save_path = os.path.join(current_experiment_dir, f"best_model.pth")
    epoch_log_path = os.path.join(current_experiment_dir, f"epoch_log_{timestamp}.csv")
    test_report_path = os.path.join(current_experiment_dir, f"test_classification_report_ALL_HEADS_{timestamp}.txt")
    test_metrics_csv_path = os.path.join(current_experiment_dir, f"test_summary_metrics_ALL_HEADS_{timestamp}.csv")

    epoch_csv_header = [
        "Epoch", "LR", "Avg_Total_Loss",
        "Avg_CE_Final", "Avg_CE_Aux", "Avg_KL_Distill", "Avg_L2_Hint",
        "Val_Accuracy_FinalHead"
    ]
    with open(epoch_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_csv_header)

    # --- 2. Data Preparation ---
    print("\n--- Starting Data Preparation ---")
    if not download_indian_pines():
        print("Dataset download/check failed. Exiting.")
        return
    X_train, y_train, X_val, y_val, X_test, y_test = get_prepared_data()
    print(f"Number of classes determined: {config.NUM_CLASSES_ACTUAL}")
    if config.NUM_CLASSES_ACTUAL is None or config.NUM_CLASSES_ACTUAL <=0:
        raise ValueError("Number of classes not determined correctly from data_preprocessing.py.")

    # --- 3. Create DataLoaders ---
    print("\n--- Creating DataLoaders ---")
    train_dataset = HSIDataset(X_train, y_train)
    val_dataset = HSIDataset(X_val, y_val) if X_val.shape[0] > 0 else None
    test_dataset = HSIDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader) if val_loader else 0} batches. Test loader: {len(test_loader)} batches.")

    # --- 4. Initialize Model, Optimizer, Scheduler ---
    print("\n--- Initializing Model and Optimizer (HybridSN-inspired) ---")
    model = HybridSNwithDistillation( # MODIFIED: Use HybridSN model
        input_bands=config.NUM_PCA_COMPONENTS, # From config (was HYBRIDSN_INPUT_BANDS)
        patch_size_spatial=config.PATCH_SIZE,  # From config (was HYBRIDSN_PATCH_SIZE)
        num_classes=config.NUM_CLASSES_ACTUAL
    ).to(device)

    # Optimizer and Scheduler: Values from config.py will be used.
    # Ensure config.LEARNING_RATE and config.WEIGHT_DECAY are appropriate for HybridSN.
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY) # Alternative
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    try:
        from torchinfo import summary
        summary_batch_size = config.BATCH_SIZE if config.BATCH_SIZE > 0 else 1
        # Input for HybridSN is (B, Bands=NUM_PCA_COMPONENTS, H=PATCH_SIZE, W=PATCH_SIZE)
        summary(model, input_size=(summary_batch_size, config.NUM_PCA_COMPONENTS, config.PATCH_SIZE, config.PATCH_SIZE), verbose=0)
        print("Model summary generated.")
    except Exception as e:
        print(f"torchinfo summary failed or not installed: {e}")

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_accuracy = 0.0
    best_epoch = -1

    for epoch in range(1, config.EPOCHS + 1):
        avg_loss, avg_components = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config.EPOCHS
        )
        current_lr = scheduler.get_last_lr()[0]
        log_str = (f"Epoch [{epoch}/{config.EPOCHS}] | LR: {current_lr:.2e} | "
                   f"Avg Loss: {avg_loss:.4f} (CE_F: {avg_components.get('ce_final',0):.3f}, "
                   f"CE_A: {avg_components.get('ce_aux_sum',0):.3f}, KL: {avg_components.get('kl_distill_sum',0):.3f}, "
                   f"L2_H: {avg_components.get('l2_hint_sum',0):.4f})")

        current_val_accuracy_final_head = -1.0 
        if val_loader:
            val_accuracy_final_head, _, _ = evaluate(model, val_loader, device, epoch, config.EPOCHS, 
                                                     eval_type="Validation", evaluate_all_heads=False)
            log_str += f" | Val Acc (Final Head): {val_accuracy_final_head:.2f}%"
            current_val_accuracy_final_head = val_accuracy_final_head
        else: 
            log_str += " | No validation set."
        print(log_str)

        epoch_data_row = [
            epoch, f"{current_lr:.2e}", f"{avg_loss:.4f}",
            f"{avg_components.get('ce_final',0):.4f}", f"{avg_components.get('ce_aux_sum',0):.4f}",
            f"{avg_components.get('kl_distill_sum',0):.4f}", f"{avg_components.get('l2_hint_sum',0):.4f}",
            f"{current_val_accuracy_final_head:.2f}" if current_val_accuracy_final_head != -1.0 else "N/A"
        ]
        with open(epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_data_row)
        scheduler.step()

        if current_val_accuracy_final_head > best_val_accuracy:
            best_val_accuracy = current_val_accuracy_final_head
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"  New best validation accuracy (Final Head): {best_val_accuracy:.2f}%. Model saved to {model_save_path}")
        elif not val_loader and epoch == config.EPOCHS:
             torch.save(model.state_dict(), model_save_path)
             print(f"  No validation. Saved model from final epoch to {model_save_path}")

    print("\n--- Training Finished ---")
    if best_epoch != -1: print(f"Best Validation Accuracy (Final Head): {best_val_accuracy:.2f}% at Epoch {best_epoch}")
    else: print("Training complete.")

    # --- 6. Evaluation on Test Set (All Heads) ---
    print("\n--- Evaluating on Test Set with Best Model (All Heads) ---")
    # ... (The rest of the evaluation and CSV/report saving logic for ALL HEADS remains the same as your previous main.py) ...
    # ... (It correctly uses the 'evaluate' function with 'evaluate_all_heads=True' and processes the list of accuracies/predictions) ...
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded best model from {model_save_path}")
    else:
        print(f"Warning: Best model path {model_save_path} not found. Evaluating with the last state of the model.")

    test_accuracies_all_heads, test_preds_all_heads, test_gt_labels = evaluate(
        model, test_loader, device, eval_type="Test", evaluate_all_heads=True
    )

    print("\n--- Test Set Performance (Per Head) ---")
    with open(test_report_path, 'w') as f_report, open(test_metrics_csv_path, 'w', newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_header = ["Head_Index", "OA", "AA", "Kappa"]
        class_names_for_header = [f"Class {i}" for i in range(config.NUM_CLASSES_ACTUAL)]
        for cn_idx, cn in enumerate(class_names_for_header): # Use cn_idx for safety if class names are not just "Class i"
            csv_header.extend([f"{cn}_precision", f"{cn}_recall", f"{cn}_f1-score", f"{cn}_support"])
        csv_writer.writerow(csv_header)

        for head_idx in range(len(test_accuracies_all_heads)):
            accuracy_current_head = test_accuracies_all_heads[head_idx]
            preds_current_head = test_preds_all_heads[head_idx]
            header_text = f"--- Metrics for Head {head_idx+1} (0-indexed: {head_idx}) {'(FINAL)' if head_idx == len(test_accuracies_all_heads) -1 else ''} ---"
            print(header_text); f_report.write(header_text + "\n")
            print(f"  Overall Accuracy (OA): {accuracy_current_head:.2f}%"); f_report.write(f"  Overall Accuracy (OA): {accuracy_current_head:.2f}%\n")

            oa_val = accuracy_current_head / 100.0; aa_val = 0.0; kappa_val = 0.0
            per_class_metrics_for_csv_current_head = {}

            if len(test_gt_labels) > 0 and len(preds_current_head) > 0 and len(test_gt_labels) == len(preds_current_head):
                class_names_report_sklearn = [f"Class {i}" for i in range(config.NUM_CLASSES_ACTUAL)]
                try:
                    report_text = classification_report(test_gt_labels, preds_current_head, target_names=class_names_report_sklearn, labels=np.arange(config.NUM_CLASSES_ACTUAL), digits=4, zero_division=0)
                    print(report_text); f_report.write(report_text + "\n")
                    kappa_val = cohen_kappa_score(test_gt_labels, preds_current_head)
                    print(f"  Cohen's Kappa: {kappa_val:.4f}"); f_report.write(f"  Cohen's Kappa: {kappa_val:.4f}\n")
                    cm = confusion_matrix(test_gt_labels, preds_current_head, labels=np.arange(config.NUM_CLASSES_ACTUAL))
                    class_accuracies = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6) 
                    aa_val = np.mean(class_accuracies[~np.isnan(class_accuracies)])
                    print(f"  Average Accuracy (AA): {aa_val:.4f}\n"); f_report.write(f"  Average Accuracy (AA): {aa_val:.4f}\n\n")

                    report_dict = classification_report(test_gt_labels, preds_current_head, target_names=class_names_report_sklearn, labels=np.arange(config.NUM_CLASSES_ACTUAL), digits=4, zero_division=0, output_dict=True)
                    for class_label_str in report_dict: # Keys are "Class 0", "Class 1", etc. or "accuracy", "macro avg"
                        if class_label_str in class_names_report_sklearn: # Process only class-specific dicts
                            for metric in ['precision', 'recall', 'f1-score', 'support']:
                                per_class_metrics_for_csv_current_head[f"{class_label_str}_{metric}"] = report_dict[class_label_str][metric]
                except Exception as e: print(f"  Error generating report for head {head_idx+1}: {e}"); f_report.write(f"  Error: {e}\n\n")
            
            csv_data_row = [f"Head_{head_idx+1}", f"{oa_val:.4f}", f"{aa_val:.4f}", f"{kappa_val:.4f}"]
            for cn_header in class_names_for_header:
                for metric in ['precision', 'recall', 'f1-score', 'support']:
                    val = per_class_metrics_for_csv_current_head.get(f'{cn_header}_{metric}', 0) # Use .get for safety
                    csv_data_row.append(f"{val:.4f}" if isinstance(val, float) else val)
            csv_writer.writerow(csv_data_row)

    print(f"\nTest summary metrics (per head) saved to: {test_metrics_csv_path}")
    print(f"Epoch log saved to: {epoch_log_path}")
    print(f"Full test report (per head) saved to: {test_report_path}")
    print(f"Best model saved to: {model_save_path}")
    print("\n--- Experiment Finished ---")

if __name__ == '__main__':
    main_experiment()