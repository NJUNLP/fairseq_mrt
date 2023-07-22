data_bin_path=data-bin/wmt14_en2de_cased
baseline_model_path=${data_bin_path}/checkpoint.sacrebleu_28.4.pt
ckpts_path=checkpoints/wmt14_en2de_bleurt_beam12_lr5e-4
metric=BLEURT
beam=4
lenpen=0.6
lang=en2de

cp $baseline_model_path $ckpts_path
mv $ckpts_path/$baseline_model $ckpts_path/checkpoint_1_0.pt
generate_path=$ckpts_path/analysis
mkdir -p $generate_path
ckpts=$(find $ckpts_path -name 'checkpoint_[0-9]_*.pt')


len=0
ed=0

echo "#=========================== generate hypo ===========================#"
cp $ckpts_path/fairseq_train.log $generate_path
for ckpt in $ckpts;do
    temp=${ckpt##*_}
    step=${temp%.*}
    len=$(($len+1))
    if [[ ${ed} -lt $step ]];then 
        ed=$step
    fi
    fairseq-generate \
        $data_bin_path \
        --path $ckpt \
        --results-path $generate_path/generate_${step}_beam${beam} \
        --batch-size 128 \
        --tokenizer moses \
        --beam $beam \
        --remove-bpe \
        --gen-subset test \
        --lenpen $lenpen
    python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
    python3 mrt_scripts/analysis/hypo_freq_stat_command.py --generate_prefix $generate_path/generate_${step}_beam${beam}
done

# TODO: prepare metric model

echo "#=========================== cal bleu ===========================#"
python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path

echo "#=========================== cal bertscore ===========================#"
python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path

echo "#=========================== cal bartscore ===========================#"
python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path

echo "#=========================== cal bleurt ===========================#"
python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path

echo "#=========================== cal comet ===========================#"
python3 mrt_scripts/metrics_test/comet_test/cal_comet_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path

echo "#=========================== cal unite ===========================#"
python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info ref --generate_path $generate_path
python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info src_ref --generate_path $generate_path

echo "#=========================== plot metrics ===========================#"
python3 mrt_scripts/analysis/plot_metrics_change_command.py --len $len --ed $ed --lang $lang --this_metric $metric --generate_prefix $generate_path
