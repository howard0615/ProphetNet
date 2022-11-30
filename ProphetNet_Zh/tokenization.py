from transformers import BertTokenizer

def prophetnet_zh_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer("prophetnet_chinese_dict/vocab_for_huggingface.txt")
    for line in fin:
        word_pieces = tok.tokenize(line.strip())
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))

prophetnet_zh_tokenize('data/src_tgt/train.src', 'data/tokenized/tokenized_train.src')
prophetnet_zh_tokenize('data/src_tgt/train.tgt', 'data/tokenized/tokenized_train.tgt')
prophetnet_zh_tokenize('data/src_tgt/valid.src', 'data/tokenized/tokenized_valid.src')
prophetnet_zh_tokenize('data/src_tgt/valid.tgt', 'data/tokenized/tokenized_valid.tgt')
prophetnet_zh_tokenize('data/src_tgt/test.src', 'data/tokenized/tokenized_test.src')
prophetnet_zh_tokenize('data/src_tgt/test.tgt', 'data/tokenized/tokenized_test.tgt')
