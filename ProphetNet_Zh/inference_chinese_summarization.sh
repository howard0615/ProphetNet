#!/bin/bash  
# matinf-summ
i=best
BEAM=4
LENPEN=1.0
SUBSET=test
DATA_DIR=./data/processed/

# SUFFIX=_summarization_beam${BEAM}_lp${LENPEN}_${SUBSET}_ck${i}
# CHECK_POINT=./models/finetune_summarization/checkpoint_${i}.pt
# OUTPUT_FILE=outputs/output${SUFFIX}.txt


# PYTHONIOENCODING=utf8 python /workplace/yhcheng/summarization_zh/ProphetNet/fairseq/generate.py ${DATA_DIR} --path ${CHECK_POINT} --user-dir ./prophetnet --task translation_prophetnet --batch-size 64 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

# grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3-  | sed "s/ ##//g" > outputs/sort_hypo${SUFFIX}.txt

for idx in 1 2 3 4 5 6 7 8 9
do
    SUFFIX=_summarization_beam${BEAM}_lp${LENPEN}_${SUBSET}_ck${idx}
    CHECK_POINT=./models/finetune_summarization/checkpoint${idx}.pt
    OUTPUT_FILE=outputs/output${SUFFIX}.txt

    PYTHONIOENCODING=utf8 fairseq-generate ${DATA_DIR} --path ${CHECK_POINT} --user-dir ./prophetnet --task translation_prophetnet --batch-size 512 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

    grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3-  | sed "s/ ##//g" > outputs/sort_hypo${SUFFIX}.txt
done