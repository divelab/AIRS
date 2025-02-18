This is the official implement of Paper [Learning to Discover Regulatory Elements for Gene Expression Prediction]().


$Cell_Type=K562 or GM12878  
$DATA_ROOT is the data root path  

To get the results of Seq2Exp
```bash
CUDA_VISIBLE_DEVICES=0 python -m train experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group=CAGE_$Cell_Type_bimambaRNP_MI \
    wandb.name=sigall_sigscale10_kl001_r01_evalsoft \
    hydra.run.dir="./outputs/gene_exp_CAGE_$Cell_Type_bimambaRNP/sigall_sigscale10_kl001_r01_evalsoft" \
    train.single_CV=11 \
    dataset.expr_type=CAGE \
    dataset.cell_type=$Cell_Type \
    model="gene_express_bimamba_MI_RNP" \
    task="extract_rationale" \
    task.loss.kl_loss_weight=0.01 \
    model.config.prior_scale_factor=10.0 \
    model.config.marginal_mean=0.1 \
    model.config.beta_min=1 \
    dataset.data_folder=$DATA_ROOT
```

To reproduce the results of EPInformer, run the following
```bash
CUDA_VISIBLE_DEVICES=0 python -m train experiment=hg38/CAGE_pred_promo_enhan_inter \
  wandb.mode=online \
  wandb.group=CAGE_pe_inter_$Cell_Type \
  wandb.name=$Cell_Type_CAGE_feat3 \
  hydra.run.dir="./outputs/$Cell_Type_CAGE_pred_pe/feat3" \
  dataset.expr_type=CAGE \
  dataset.cell_type=$Cell_Type \
  dataset.n_extraFeat=3 \
  dataset.data_folder=$DATA_ROOT
```

To reproduce the results of Enformer, run the following
```bash
CUDA_VISIBLE_DEVICES=3 python -m train experiment=hg38/gene_express \
  wandb.mode=online \
  wandb.group=Enformer_CAGE_$Cell_Type \
  wandb.name=pretrainFalse \
  hydra.run.dir="./outputs/Enformer_CAGE_$Cell_Type/pretrainFalse" \
  dataset.expr_type=CAGE \
  dataset.cell_type=$Cell_Type \
  model="Enformer" \
  task="gene_pred" \
  model.config.use_pretrain=False \
  dataset.batch_size=4 \
  dataset.tokenizer_name=char \
  optimizer.lr=7e-6 \
  trainer.max_steps=200000 \
  scheduler.warmup_lr_init=0.0 \
  scheduler.warmup_t=50000 \
  scheduler.lr_min=7e-6 \
  dataset.data_folder=$DATA_ROOT
```

The dataset is huge, and we will release all the raw & processed data, and model weights in the future.
