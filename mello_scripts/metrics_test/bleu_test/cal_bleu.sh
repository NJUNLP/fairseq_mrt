ref=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5_fi2en/generate_fi2en_v3_beam10/ref.txt
hyp=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5_fi2en/generate_fi2en_v3_beam10/hyp.txt
sacrebleu $ref -i $hyp -l fi-en -b



# bash /opt/tiger/fake_arnold/fairseq_mrt/mello_scripts/metrics_test/bleu_test/cal_bleu.sh
