"""Class registry for models, layers, optimizers, and schedulers.

"""

optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Pre-training LM head models
    "hyena_lm": "src.models.sequence.long_conv_lm.ConvLMHeadModel",
    "mamba_lm": "mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel",
    "caduceus_lm": "caduceus.modeling_caduceus.CaduceusForMaskedLM",

    # Downstream task embedding backbones
    "dna_embedding": "src.models.sequence.dna_embedding.DNAEmbeddingModel",
    "dna_embedding_mamba": "src.models.sequence.dna_embedding.DNAEmbeddingModelMamba",
    "dna_embedding_caduceus": "src.models.sequence.dna_embedding.DNAEmbeddingModelCaduceus",

    # Baseline for genomics benchmark
    "genomics_benchmark_cnn": "src.models.baseline.genomics_benchmark_cnn.GenomicsBenchmarkCNN",

    # generation model
    "transformer_lm": "src.models.sequence.gpt_model.GPT",
    "bert_lm": "src.models.sequence.bert_model.Bert",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "ff": "src.models.sequence.ff.FF",
    "hyena": "src.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src.models.sequence.hyena.HyenaFilter",
}

callbacks = {
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_n_steps": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_epoch": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "params": "src.callbacks.params.ParamsLog",
    "timer": "src.callbacks.timer.Timer",
    "val_every_n_global_steps": "src.callbacks.validation.ValEveryNGlobalSteps",
}

model_state_hook = {
    'load_backbone': 'src.models.sequence.dna_embedding.load_backbone',
}
