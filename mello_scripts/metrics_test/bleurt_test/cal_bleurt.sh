fairseq_generate_prefix=/opt/tiger/fake_arnold/fairseq_mrt_wangtao/checkpoints/wmt14_en2de_uncased_bleurt_beam12_lr5e-4/generate_checkpoint_1_30.pt_beam4
hyp=$fairseq_generate_prefix/hyp.txt
ref=$fairseq_generate_prefix/ref.txt

python -m bleurt.score_files \
  -candidate_file=$hyp \
  -reference_file=$ref \
  -bleurt_checkpoint=BLEURT-20

