len=2
ed=50
beam=10
lang=en2zh
generate_path=/opt/tiger/fake_arnold/fairseq_mrt/mrt_analysis_results/en2zh_comet
python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path

# bash /opt/tiger/fake_arnold/fairseq_mrt/mrt_scripts/tool/cal_bartscore_command.sh
