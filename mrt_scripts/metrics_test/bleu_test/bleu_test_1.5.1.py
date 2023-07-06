import sacrebleu
import os

# hyp = 'Die Wahrheit ist jedoch kein Verbrechen.'
# ref = '1'
# bleu_score = sacrebleu.corpus_bleu([hyp], [[ref]]).score
# print(bleu_score)


refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)

bleu_list = []
for h, y in zip(sys, refs[0]):
    bleu_list.append(sacrebleu.corpus_bleu([h], [[y]]).score)
    print(sacrebleu.corpus_bleu([h], [[y]]).score)
print(sum(bleu_list) / len(bleu_list))

lang = 'en-de'
ref = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5/generate_beam4_v6/ref.txt'
hyp = '/opt/tiger/fake_arnold/fairseq_mrt/checkpoints/average_5/generate_beam4_v6/hyp.txt'
command = 'sacrebleu ' + ref + ' -i ' + hyp + ' -l ' + lang + ' -b'
a = os.popen(command)
score = float(a.readline())
print(score)

# python3 mrt_scripts/metrics_test/bleu_test/bleu_test_1.5.1.py
