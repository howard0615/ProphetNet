from rouge.rouge import FilesRouge

hyps_dir = "/workplace/yhcheng/summarization_zh/ProphetNet/ProphetNet_Zh/outputs/sort_hypo_summarization_beam4_lp1.0_test_ckbest.txt"
refs_dir = "/workplace/yhcheng/summarization_zh/ProphetNet/ProphetNet_Zh/data/src_tgt/tokenized_test.tgt"

def main():
    rouge = FilesRouge(hyp_path=hyps_dir, ref_path=refs_dir)
    try:
        rouge_summ = rouge.get_scores(avg=True, ignore_empty=True)
        print(rouge_summ)
    except RuntimeError:
        print('Failed to compute Rouge!')

if __name__ == "__main__":
    main()
