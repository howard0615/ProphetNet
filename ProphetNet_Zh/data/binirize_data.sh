fairseq-preprocess \
--user-dir ../prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref src_tgt/tokenized_train --validpref src_tgt/tokenized_valid --testpref src_tgt/tokenized_test \
--destdir processed --srcdict ../prophetnet_chinese_dict/vocab_for_fairseq.txt --tgtdict ../prophetnet_chinese_dict/vocab_for_fairseq.txt \
--workers 20