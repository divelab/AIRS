DATA_ROOT=${1}
Cell_Type=${2:-GM12878}

if [ -z "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT is not provided."
    exit 1
fi


# Seq2Exp-soft
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_bimambaRNP_MI" \
    wandb.name=sigall_sigscale10_kl001_r01_evalsoft \
    hydra.run.dir="./outputs/gene_exp_CAGE_${Cell_Type}_bimambaRNP/sigall_sigscale10_kl001_r01_evalsoft" \
    train.single_CV=11 \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_bimamba_MI_RNP" \
    task="extract_rationale" \
    task.loss.kl_loss_weight=0.01 \
    model.config.prior_scale_factor=10.0 \
    model.config.marginal_mean=0.1 \
    model.config.beta_min=1 \
    dataset.data_folder="$DATA_ROOT"


# Seq2Exp-hard
python -m train \
    experiment=hg38/gene_express \
    wandb.mode=online \
    wandb.group="CAGE_${Cell_Type}_bimambaRNP_MI" \
    wandb.name=sigall_sigscale1_r045_thres05_scale01_bmin0_trainevalhard_pspos \
    hydra.run.dir="./outputs/gene_exp_CAGE_${Cell_Type}_bimambaRNP/sigall_sigscale1_r045_thres05_scale01_bmin0_trainevalhard_pspos" \
    train.single_CV=11 \
    dataset.expr_type=CAGE \
    dataset.cell_type="$Cell_Type" \
    model="gene_express_bimamba_MI_RNP" \
    task="extract_rationale" \
    task.loss.kl_loss_weight=0.01 \
    model.config.prior_scale_factor=1.0 \
    model.config.beta_min=0 \
    model.config.marginal_mean=0.45 \
    model.config.marginal_scale=0.1 \
    model.config.z_scale=1.0 \
    model.config.test_soft=False \
    model.config.test_hard=True \
    model.config.post_hard_dist=True \
    model.config.enc_prd_ps=True \
    model.config.pos_enc=True \
    model.config.sample_threshold=0.5 \
    dataset.data_folder="$DATA_ROOT"
