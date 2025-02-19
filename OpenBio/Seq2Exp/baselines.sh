DATA_ROOT=${1}
Cell_Type=${2:-GM12878}

if [ -z "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT is not provided."
    exit 1
fi


# Enformer
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="Enformer_CAGE_${Cell_Type}" \
    wandb.name=pretrainFalse \
    hydra.run.dir="./outputs/Enformer_CAGE_${Cell_Type}/pretrainFalse" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
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
    dataset.data_folder="$DATA_ROOT"


# hyena
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_seq_model" \
    wandb.name="CAGE_${Cell_Type}_hyena_feat0_seq200k_layer4" \
    hydra.run.dir="./outputs/gene_exp/CAGE_${Cell_Type}_hyena_feat0_seq200k_layer4" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_hyena" \
    dataset.batch_size=8 \
    dataset.seq_range=200_000 \
    model.n_layer=4 \
    dataset.tokenizer_name='char' \
    train.single_CV=11 \
    dataset.data_folder="$DATA_ROOT"


# mamba
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_seq_model" \
    wandb.name="CAGE_${Cell_Type}_mamba_feat0_seq200k_layer4" \
    hydra.run.dir="./outputs/gene_exp/CAGE_${Cell_Type}_mamba_feat0_seq200k_layer4" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_mamba" \
    dataset.batch_size=16 \
    dataset.seq_range=200_000 \
    model.config.n_layer=4 \
    dataset.tokenizer_name='char' \
    train.single_CV=11 \
    dataset.data_folder="$DATA_ROOT"


# caduceus
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_seq_model" \
    wandb.name="CAGE_${Cell_Type}_bimamba_feat0_seq200k_layer4" \
    hydra.run.dir="./outputs/gene_exp/CAGE_${Cell_Type}_bimamba_feat0_seq200k_layer4" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_bi_mamba" \
    dataset.batch_size=16 \
    dataset.seq_range=200_000 \
    model.config.n_layer=4 \
    train.single_CV=11 \
    model.config.interact=no_signal \
    dataset.data_folder="$DATA_ROOT"


# caduceus with signals
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_seq_model" \
    wandb.name="CAGE_${Cell_Type}_bimamba_feat3_seq200k_layer4" \
    hydra.run.dir="./outputs/gene_exp/CAGE_${Cell_Type}_bimamba_feat3_seq200k_layer4" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_bi_mamba" \
    dataset.batch_size=16 \
    model.config.n_layer=4 \
    dataset.data_folder="$DATA_ROOT"


# EPInformer
python -m train \
    experiment=hg38/CAGE_pred_promo_enhan_inter \
    wandb.mode=online \
    wandb.group="CAGE_pe_inter_${Cell_Type}" \
    wandb.name="${Cell_Type}_CAGE_feat3" \
    hydra.run.dir="./outputs/${Cell_Type}_CAGE_pred_pe/feat3" \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    dataset.n_extraFeat=3 \
    dataset.data_folder="$DATA_ROOT"

