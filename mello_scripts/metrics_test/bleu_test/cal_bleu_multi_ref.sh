ref1=/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en0
ref2=/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en1
ref3=/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en2
ref4=/opt/tiger/fake_arnold/fairseq_mrt/data/nist_test/MT040506/en3

hyp=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_10/generate_zh2en_baseline_beam4/hyp.txt
sacrebleu $ref1 $ref2 $ref3 $ref4 -i $hyp -l zh-en -b



# bash /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/metrics_test/bleu_test/cal_bleu_multi_ref.sh
