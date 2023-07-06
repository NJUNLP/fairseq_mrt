cmd1=mrt/en2de/data-bin/wmt14_en2de_cased
data_bin_path=data-bin/${cmd1##*/}

cmd2=mrt/en2de/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters
ckpts_path=checkpoints/${cmd2##*/}
analysis_path=$ckpts_path/analysis
generate_path=$analysis_path/generate_hypo
mkdir -p $generate_path

baseline_model_path=$(find data-bin -name '*.pt')
baseline_model=${baseline_model_path##*/}

# cp $baseline_model_path $ckpts_path
# mv $ckpts_path/$baseline_model $ckpts_path/checkpoint_1_0.pt

# 循环generate hypo
ckpts=$(find $ckpts_path -name 'checkpoint_[0-9]_*.pt')

beam=4

echo "#=========================== generate hypo ===========================#"
step_array=()

for ckpt in $ckpts;do
    echo $ckpt
    temp=${ckpt##*_}
    step=${temp%.*}
    echo $step
    step_array[${#step_array[*]}]=$step
    fairseq-generate \
        $data_bin_path \
        --path $ckpt \
        --results-path $generate_path/generate_${step}_beam${beam} \
        --max-tokens 1000 \
        --tokenizer moses \
        --beam $beam \
        --max-len-a 1.2 --max-len-b 10 \
        --remove-bpe \
        --gen-subset valid
    python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
done


# for ckpt in $ckpts;do
#     #echo $ckpt
#     temp=${ckpt##*_}
#     step=${temp%.*}
#     #echo $step
#     step_array[${#step_array[*]}]=$step
# done
sorted_step_array=(1 2)
step_array=(50 0 100)
echo "原数组的顺序为：${step_array[@]}"
list=$(echo ${step_array[@]} | tr ' ' '\n' | sort -n)

a=0
for i in $list
do
    sorted_step_array[$a]=$i
    let a++
done
echo "新数组的顺序为：${sorted_step_array[@]}"

sorted_step_array=$(echo ${step_array[*]} | tr ' ' '\n' | sort -n)
sorted_step_array=$(cat ${step_array[*]} | tr ' ' '\n' | sort -n)
len=${#sorted_step_array[*]}
lastIndex=$((${#sorted_step_array[@]}-1))
st=${sorted_step_array[0]}
ed=${sorted_step_array[lastIndex]}
echo ${step_array[*]}
echo ${sorted_step_array[*]}
echo $len $st $ed
# ================================= # 
step_array=(50 0 100)
echo "step_array"
echo ${step_array[*]}
sorted_step_array=()
list=$(echo ${step_array[*]} | tr ' ' '\n' | sort -n)
a=0
for i in $list
do
    sorted_step_array[$a]=$i
    let a++
done

len=${#sorted_step_array[*]}
lastIndex=$((${#sorted_step_array[@]}-1))
st=${sorted_step_array[0]}
ed=${sorted_step_array[lastIndex]}
echo "sorted_step_array  len  st  ed"
echo ${sorted_step_array[*]}
echo $len
echo $st
echo $ed
# ================================= # 

#echo ${sorted_step_array[*]}

# temp
# generate_path=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters
sorted_step_array=(0 50 100)
echo ${sorted_step_array[*]}
len=${#sorted_step_array[*]}
lastIndex=$((${#sorted_step_array[@]}-1))
st=${sorted_step_array[0]}
ed=${sorted_step_array[lastIndex]}
echo $len $st $ed


# for step in ${sorted_step_array[*]};do
#     python3 mrt_scripts/analysis/split_hyp_from_fairseq_generate_command.py --prefix $generate_path/generate_${step}_beam${beam}
# done

work="generate_hypo cal_bleu cal_bertscore cal_bartscore cal_bleurt cal_comet cal_unite"

# cal bleu
echo "#=========================== cal BLEU ===========================#"
python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py --st $st --ed $ed --len $len --beam $beam --generate_path $generate_path

work="cal_bleu_a"
generate_path=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters

if [ ! -f "$generate_path/stat_bleu_1.5.1.csv" ]
then
    echo "#=========================== cal BLEU ===========================#"
fi

# ============== # 

len=0
ed=0

cmd2=mrt/en2de/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters
ckpts_path=checkpoints/${cmd2##*/}

ckpts=$(find $ckpts_path -name 'checkpoint_[0-9]_*.pt')

generate_path=/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters

if [ "`ls -A $generate_path`" = "" ]
then
    echo "balabala"
else
    for ckpt in $ckpts;do
        temp=${ckpt##*_}
        step=${temp%.*}
        # let len++
        len=$(($len+1))
        if [[ ${ed} -lt $step ]];then 
            ed=$step
        fi
    done
fi
echo "ed len"
echo $ed
echo $len


# for i in ${indexlist[@]};do 
#     if [[ ${minDate} -lt $i ]];then 
#         maxDate=$i
#     fi 
# done 

# ============== # 

cmd1=mrt/en2de/data-bin/wmt14_en2de_cased
temp=${cmd1#*/}
echo $temp
lang=${temp%%/*}
echo $lang


ckpts_path=/home/byte_arnold_lq_mlnlc/user/yanyiming.mello/mrt/de2en/checkpoints/wmt14_de2en_cased_unite_src_ref_beam12_lr5e-4_base_bleu30.21
temp=${ckpts_path#*unite_}
echo $temp
info=${temp%%_beam*}
echo $info

python3 mrt_scripts/analysis/plot_metrics_change_command.py --ed 100 --len 3 --lang en2de --this_metric BLEURT --generate_prefix checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters

cmd2=mrt/en2de/checkpoints/wmt14_en2de_cased_bleurt_beam12_lr5e-4_base_bleu26.11_continue_meters_toy
temp=${cmd2%%_beam*}
echo $temp
metric=${temp##*_}
echo $metric

# bash mrt_scripts/tool/bash_test.sh