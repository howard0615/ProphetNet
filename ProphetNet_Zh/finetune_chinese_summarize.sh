DATA_DIR=/workplace/yhcheng/summarization_zh/ProphetNet/ProphetNet_Zh/data/processed/
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=models/finetune_summarization
TENSORBOARD_LOGDIR=models/tensorboard_summarization
USER_DIR=./prophetnet
PRETRAINED_CHECKPOINT=./pretrained_model/prophetnet_zh.pt

fairseq-train \
	--fp16 --ngram 2 \
	--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.0001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion $CRITERION --label-smoothing 0.1 \
	--update-freq 4 --max-sentences 6 \
	--num-workers 8  \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--truncate-source --load-from-pretrained-model $PRETRAINED_CHECKPOINT \
	--save-dir $SAVE_DIR \
	--keep-last-epochs 10 \
	--tensorboard-logdir $TENSORBOARD_LOGDIR  \
	$DATA_DIR