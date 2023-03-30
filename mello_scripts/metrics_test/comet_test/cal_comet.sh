data_prefix=/opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/metrics_test/toy_data
src=$data_prefix/src.de
hyp=$data_prefix/hyp1.en
ref=$data_prefix/ref.en

# comet-score -s $src -t $hyp -r $ref --quiet
python3 comet/cli/score.py -s $src -t $hyp -r $ref --quiet  # 如果是本地下载的话，就直接调用python文件

# comet-score -s $src -t $hyp -r $ref --gpus 8 --quiet  # 多卡比单卡还慢好像？

# bash mello_scripts/metrics_test/comet_test/cal_comet.sh