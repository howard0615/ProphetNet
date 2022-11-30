import pandas as pd

# datasets : MATINF-summ

matinf_dir = "/workplace/yhcheng/summarization_zh/datasets/MATINF-summ"
src_tgt_dir = "src_tgt"

def get_src_tgt(dataframe):
    """ id,question,description,answer,class """
    source_list = dataframe.description.to_list()
    target_list = dataframe.question.to_list()
    src = "\n".join(source_list)
    tgt = "\n".join(target_list)
    return src, tgt

def write_to_file(mode, src, tgt):
    with open(src_tgt_dir+"/{}.src".format(mode), "w", encoding="utf-8")as s, open(src_tgt_dir+"/{}.tgt".format(mode), "w", encoding="utf-8")as t:
        s.write(src)
        t.write(tgt)

def main():
    
    train_df = pd.read_csv(matinf_dir + "/train.csv")
    dev_df = pd.read_csv(matinf_dir + "/dev.csv")
    test_df = pd.read_csv(matinf_dir + "/test.csv")

    train_src, train_tgt = get_src_tgt(train_df)
    dev_src, dev_tgt = get_src_tgt(dev_df)
    test_src, test_tgt = get_src_tgt(test_df)

    write_to_file("train", train_src, train_tgt)
    write_to_file("valid", dev_src, dev_tgt)
    write_to_file("test", test_src, test_tgt)


if __name__=="__main__":
    main()
