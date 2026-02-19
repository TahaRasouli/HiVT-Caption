import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score

class ManeuverGRU(pl.LightningModule):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=5, lr=1e-3, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.num_classes = num_classes
        
        if weights is not None:
            self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float))
        else:
            self.loss_weights = None

        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
        self.class_names = ["Maintain", "Turn L", "Turn R", "LC L", "LC R"]
        
        # Metrics
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision_per_class = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        
        # NEW: Macro F1 for monitoring the "best" model
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), y.view(-1), weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1).view(-1)
        targets = y.view(-1)
        
        mask = targets != -100
        f_preds, f_targets = preds[mask], targets[mask]

        loss = F.cross_entropy(logits.view(-1, self.num_classes), targets, weight=self.loss_weights)
        
        # Update all metrics
        self.val_acc(f_preds, f_targets)
        self.val_precision_per_class(f_preds, f_targets)
        self.val_f1_per_class(f_preds, f_targets)
        self.val_f1_macro(f_preds, f_targets)
        
        self.log("val_loss", loss, prog_bar=True)
        # Log macro F1 so Checkpoint can see it
        self.log("val_f1_macro", self.val_f1_macro, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        precisions = self.val_precision_per_class.compute()
        f1_scores = self.val_f1_per_class.compute()
        macro_f1 = self.val_f1_macro.compute()
        
        print(f"\n{'='*60}\nEpoch {self.current_epoch} Metrics (Macro F1: {macro_f1:.4f})\n{'-'*60}")
        print(f"{'Class Name':<20} | {'Precision':<10} | {'F1-Score':<10}")
        print(f"{'-'*60}")
        for i, name in enumerate(self.class_names):
            print(f"{name:<20} | {precisions[i]:.4f}     | {f1_scores[i]:.4f}")
        print(f"{'-'*60}\nTotal Val Accuracy: {acc:.4f}\n{'='*60}\n")
        
        self.val_acc.reset()
        self.val_precision_per_class.reset()
        self.val_f1_per_class.reset()
        self.val_f1_macro.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class StatefulManeuverGRU(pl.LightningModule):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=5, lr=1e-3, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.num_classes = num_classes
        
        # 1. Architecture
        # Stateful GRUs are usually unidirectional to maintain true causal time flow
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 2. Weights and Buffers
        if weights is not None:
            self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float))
        else:
            self.loss_weights = None

        # 3. State Management
        self.hidden_state = None
        self.last_scene_id = None

        # 4. Metrics
        self.class_names = ["Maintain", "Turn L", "Turn R", "LC L", "LC R"]
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision_per_class = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x, h0=None):
        # out: [Batch, Seq, HiddenSize]
        out, h_n = self.gru(x, h0)
        logits = self.classifier(out)
        return logits, h_n

    def training_step(self, batch, batch_idx):
        x, y, scene_ids = batch
        
        # Identify current scene (assuming batch_size=1)
        current_scene = scene_ids[0]

        # Reset state if we hit a new scene
        if current_scene != self.last_scene_id:
            self.hidden_state = None
        
        # Forward pass
        logits, self.hidden_state = self(x, self.hidden_state)
        
        # IMPORTANT: Detach state so we don't backprop through the whole scene history
        self.hidden_state = self.hidden_state.detach()
        self.last_scene_id = current_scene

        loss = F.cross_entropy(logits.view(-1, self.num_classes), y.view(-1), weight=self.loss_weights)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, scene_ids = batch
        
        # Similar state logic for validation to ensure consistency
        current_scene = scene_ids[0]
        if current_scene != self.last_scene_id:
            self.hidden_state = None

        logits, self.hidden_state = self(x, self.hidden_state)
        self.hidden_state = self.hidden_state.detach()
        self.last_scene_id = current_scene

        preds = torch.argmax(logits, dim=-1).view(-1)
        targets = y.view(-1)
        
        mask = targets != -100
        f_preds, f_targets = preds[mask], targets[mask]

        loss = F.cross_entropy(logits.view(-1, self.num_classes), targets, weight=self.loss_weights)
        
        # Update Metrics
        self.val_acc(f_preds, f_targets)
        self.val_precision_per_class(f_preds, f_targets)
        self.val_f1_per_class(f_preds, f_targets)
        self.val_f1_macro(f_preds, f_targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        precisions = self.val_precision_per_class.compute()
        f1_scores = self.val_f1_per_class.compute()
        macro_f1 = self.val_f1_macro.compute()
        
        print(f"\n{'='*60}\nEpoch {self.current_epoch} Metrics (Macro F1: {macro_f1:.4f})\n{'-'*60}")
        print(f"{'Class Name':<20} | {'Precision':<10} | {'F1-Score':<10}")
        print(f"{'-'*60}")
        
        for i, name in enumerate(self.class_names):
            p = precisions[i].item()
            f = f1_scores[i].item()
            print(f"{name:<20} | {p:.4f}     | {f:.4f}")

        print(f"{'-'*60}\nTotal Val Accuracy: {acc:.4f}\n{'='*60}\n")
        
        # Reset
        self.val_acc.reset()
        self.val_precision_per_class.reset()
        self.val_f1_per_class.reset()
        self.val_f1_macro.reset()
        # Also clear state trackers at end of epoch
        self.hidden_state = None
        self.last_scene_id = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)