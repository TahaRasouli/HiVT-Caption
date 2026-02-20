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

    def test_step(self, batch, batch_idx):
        # We reuse the validation logic exactly
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        # We reuse the validation end logic to print the final table
        print("\n" + "="*20 + " FINAL TEST RESULTS " + "="*20)
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ManeuverLSTM(pl.LightningModule):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, num_classes=5, lr=1e-3, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        
        self.ln = nn.LayerNorm(input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        
        # --- NEW: The Residual Projection Layer ---
        # Projects input_size (9) to (hidden_size * 2) which is 256
        self.classifier_input_projection = nn.Linear(input_size, hidden_size * 2)
        
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
        if weights is not None:
            self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float))
        else:
            self.loss_weights = None

        self.class_names = ["Maintain", "Turn L", "Turn R", "LC L", "LC R"]
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)

    # Original LSTM
    # def forward(self, x):
    #     # x shape: [Batch, Seq, 9]
    #     x = self.ln(x)
        
    #     # LSTM returns: output, (h_n, c_n)
    #     # output shape: [Batch, Seq, hidden_size * 2]
    #     out, _ = self.lstm(x)
        
    #     return self.classifier(out)

    # Bidirectional LSTM
    def forward(self, x):
        # x: [Batch, Seq, 9]
        norm_x = self.ln(x)
        
        # 1. Temporal Branch: LSTM captures the racing line 'flow'
        # out: [Batch, Seq, 256]
        out, _ = self.lstm(norm_x)
        
        # 2. Skip Connection: Direct physics info
        # res: [Batch, Seq, 256]
        res = self.classifier_input_projection(norm_x) 
        
        # 3. Combine: Residual logic helps the model converge back to 0.90
        combined = out + res
        
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, 5), y.view(-1), 
                               weight=self.loss_weights, label_smoothing=0.05)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1).view(-1)
        targets = y.view(-1)
        
        mask = targets != -100
        f_preds, f_targets = preds[mask], targets[mask]

        loss = F.cross_entropy(logits.view(-1, 5), targets, weight=self.loss_weights)
        
        self.val_acc(f_preds, f_targets)
        self.val_f1_macro(f_preds, f_targets)
        self.val_f1_per_class(f_preds, f_targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        macro_f1 = self.val_f1_macro.compute()
        per_class_f1 = self.val_f1_per_class.compute()
        
        print(f"\n{'='*40}\nEpoch {self.current_epoch} - LSTM Macro F1: {macro_f1:.4f}\n{'-'*40}")
        for i, name in enumerate(self.class_names):
            print(f"{name:<12}: {per_class_f1[i]:.4f}")
        print(f"{'='*40}\n")
        
        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]


class ManeuverCNN(pl.LightningModule):
    def __init__(self, input_size=9, num_classes=5, lr=1e-3, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['weights'])
        self.lr = lr
        self.num_classes = num_classes
        
        # 1. Feature Extractor (1D-CNN)
        # Input: [Batch, 9, 50]
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 2. Point-wise Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        if weights is not None:
            self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float))
        else:
            self.loss_weights = None

        # 3. METRICS INITIALIZATION (Fixed the Missing Attributes)
        self.class_names = ["Maintain", "Turn L", "Turn R", "LC L", "LC R"]
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # Macro F1 for monitoring (The "Best" model selector)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        # Per-class metrics for the table printout
        self.val_precision_per_class = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None)

    def forward(self, x):
        # x shape: [Batch, Seq, Feats] -> e.g. [16, 50, 9]
        # Conv1d expects [Batch, Feats, Seq]
        x = x.permute(0, 2, 1)
        
        features = self.network(x) # [16, 256, 50]
        
        # Back to [Batch, Seq, 256] for the linear layer
        features = features.permute(0, 2, 1)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), y.view(-1), 
                               weight=self.loss_weights, label_smoothing=0.05)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1).view(-1)
        targets = y.view(-1)
        
        mask = targets != -100
        f_preds, f_targets = preds[mask], targets[mask]

        loss = F.cross_entropy(logits.view(-1, self.num_classes), targets, weight=self.loss_weights)
        
        # Update metrics
        self.val_acc(f_preds, f_targets)
        self.val_f1_macro(f_preds, f_targets)
        self.val_precision_per_class(f_preds, f_targets)
        self.val_f1_per_class(f_preds, f_targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1_macro", self.val_f1_macro, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute all
        acc = self.val_acc.compute()
        macro_f1 = self.val_f1_macro.compute()
        precisions = self.val_precision_per_class.compute()
        f1_scores = self.val_f1_per_class.compute()
        
        print(f"\n{'='*65}")
        print(f"Epoch {self.current_epoch} Metrics (Macro F1: {macro_f1:.4f})")
        print(f"{'-'*65}")
        print(f"{'Class Name':<20} | {'Precision':<12} | {'F1-Score':<10}")
        print(f"{'-'*65}")
        for i, name in enumerate(self.class_names):
            print(f"{name:<20} | {precisions[i]:.4f}       | {f1_scores[i]:.4f}")
        print(f"{'-'*65}\nTotal Val Accuracy: {acc:.4f}\n{'='*65}\n")
        
        # Reset all
        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_precision_per_class.reset()
        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)