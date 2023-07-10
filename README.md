# BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training

This repository contains the code for the ACL 2023 paper **BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training**.

> **The code is in the process of being organized, I'll get it done as soon as I can.**

## Universal Adversarial Translations
We find universal adversarial translations of BLEURT and BARTScore, which are capable of obtaining high scores when evaluated against any reference sentence.

An example is presented in the figure below:


<img src="figures/bleurt_universal_translation_example.png" width = "500" alt="bleurt_universal_translation_example" align=center />

 > $hypo$ means the translation sentence and $ref$ means the reference sentence. BLEURT needs to compare $hypo$ and $ref$ to judge the quality of $hypo$. This figure shows that the universal translation can achieve high BLEURT scores when calculated with each $ref$, even if $hypo$ and $ref$ are completely unrelated.


## Guide
### MRT Training
**Step1: Maximum Likelihood Estimation (MLE) training phase**

Train with conventional negative log-likelihood (NLL) loss

```
# Take En->De as example
bash mrt_scripts\fairseq_train\fairseq_train_normal_ende.sh
```

**Step 2:  MRT training phase**

Fine-tune the model with each metric, so as to obtain translation models with various metric styles

```
# Take En->De as example
bash mrt_scripts\fairseq_train\fairseq_train_mrt_ende_bleurt_beam12.sh
```

### Analyze Training Process
**Step1: Generate Hypothesis Sentences from the Training Process**
```
generate_path=$ckpts_path/analysis
mkdir -p $generate_path
ckpts=$(find $ckpts_path -name 'checkpoint_[0-9]_*.pt')

len=0
ed=0

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
```

**Step2: Calculate the Score of Each Metric**

cal BLEU
```
python3 mrt_scripts/metrics_test/bleu_test/cal_bleu_file_1.5.1_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
```

cal BERTScore
```
python3 mrt_scripts/metrics_test/bertscore_test/cal_bertscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
```

cal BARTScore
```
python3 mrt_scripts/metrics_test/bartscore_test/cal_bartscore_file_command.py --len $len --ed $ed --beam $beam --lang $lang --generate_path $generate_path
```

cal BLEURT
```
python3 mrt_scripts/metrics_test/bleurt_test/cal_bleurt_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path
```

cal COMET
```
python3 mrt_scripts/metrics_test/comet_test/cal_comet_file_command.py --len $len --ed $ed --beam $beam --generate_path $generate_path
```

cal UniTE
```
python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info ref --generate_path $generate_path
python3 mrt_scripts/metrics_test/unite_test/cal_unite_file_command.py --len $len --ed $ed --beam $beam --info src_ref --generate_path $generate_path
```

**Step3: Plot Training Process Figures**
```
python3 mrt_scripts/analysis/plot_metrics_change_command.py --len $len --ed $ed --lang $lang --this_metric $metric --generate_prefix $generate_path
```
then you can get a figure like this:
![mrt_plot_metrics_en2de](figures/mrt_plot_metrics_en2de.png "mrt_plot_metrics_en2de")
> The horizontal axis represents the training steps, and the vertical axis is the score of each metric  (except for BARTScore on the right axis, which is a negative number because it calculates the logarithmic probability of translations); metrics other than BARTScore and BLEU are mostly distributed between 0 and 1, and we multiply them uniformly by 100 for ease of observation. The asterisk represents the highest value achieved by the optimized metric.

There may be universal translations if you find a circumstance where only one metric improves while the other declines.