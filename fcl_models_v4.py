"""Model components for the isolated V4 FCL experiments.

The V4 model keeps every optional contribution behind an explicit switch so
that the ablation sequence is reproducible: adapter, SE, CosFace, replay and
DP are tested independently instead of being silently coupled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-excitation gate for the statistical feature embedding."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        bottleneck = max(4, channels // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class TaskAdapter(nn.Module):
    """Small residual adapter trained after the shared backbone is frozen."""

    def __init__(self, feature_dim, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck)
        self.up = nn.Linear(bottleneck, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        update = self.up(F.relu(self.down(x)))
        return self.norm(x + update)


class StatEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SeqEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(16 * 32, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class CosineClassifier(nn.Module):
    """Cosine classifier with non-in-place target margin application."""

    def __init__(self, feature_dim, num_classes, scale=16.0, margin=0.20):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets=None):
        normalized_features = F.normalize(features, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_features, normalized_weight).clamp(-1.0, 1.0)
        if self.training and targets is not None:
            one_hot = F.one_hot(targets, num_classes=cosine.shape[1]).to(cosine.dtype)
            cosine = cosine - one_hot * self.margin
        return self.scale * cosine


class V4DualBackbone(nn.Module):
    """Dual statistical/sequence network with independently enabled modules."""

    def __init__(
        self,
        stat_dim=77,
        seq_dim=512,
        hidden_dim=128,
        num_classes=6,
        use_stat=True,
        use_seq=True,
        use_adapter=False,
        use_se=False,
        use_cosface=False,
        use_logit_adj=False,
        class_counts=None,
        adapter_dim=32,
    ):
        super().__init__()
        if not use_stat and not use_seq:
            raise ValueError("V4DualBackbone needs at least one input tower.")
        self.use_stat = use_stat
        self.use_seq = use_seq
        self.use_adapter = use_adapter
        self.use_se = use_se
        self.use_cosface = use_cosface
        self.use_logit_adj = use_logit_adj

        self.stat_encoder = StatEncoder(stat_dim, hidden_dim) if use_stat else None
        self.se = SEBlock(hidden_dim) if use_stat and use_se else None
        self.seq_encoder = SeqEncoder(hidden_dim) if use_seq else None
        fusion_dim = hidden_dim * int(use_stat) + hidden_dim * int(use_seq)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.adapter = TaskAdapter(fusion_dim, adapter_dim) if use_adapter else None
        if use_cosface:
            self.classifier = CosineClassifier(fusion_dim, num_classes)
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)

        if class_counts is None:
            class_counts = [1] * num_classes
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        log_priors = torch.log(counts / counts.sum())
        self.register_buffer("log_priors", log_priors)

    def set_class_counts(self, class_counts):
        """Update the prior buffer for an online task-level prior."""
        counts = torch.as_tensor(class_counts, dtype=self.log_priors.dtype,
                                 device=self.log_priors.device).clamp_min(1.0)
        if counts.numel() != self.log_priors.numel():
            raise ValueError("class_counts must match the classifier output size")
        with torch.no_grad():
            self.log_priors.copy_(torch.log(counts / counts.sum()))

    def encode(self, stat_x, seq_x=None):
        features = []
        if self.use_stat:
            stat_features = self.stat_encoder(stat_x)
            if self.se is not None:
                stat_features = self.se(stat_features)
            features.append(stat_features)
        if self.use_seq:
            if seq_x is None:
                raise ValueError("Sequence features are required by this V4 model.")
            features.append(self.seq_encoder(seq_x))
        fused = self.fusion_norm(torch.cat(features, dim=1))
        if self.adapter is not None:
            fused = self.adapter(fused)
        return fused

    def forward(self, stat_x, seq_x=None, targets=None):
        features = self.encode(stat_x, seq_x)
        logits = self.classifier(features, targets) if self.use_cosface else self.classifier(features)
        if self.use_logit_adj:
            logits = logits - self.log_priors.to(logits.device)
        return logits
