from BARTScore.bart_score import BARTScorer
import numpy as np

# To use the CNNDM version BARTScore
lang = 'en-de'
bart_scorer = BARTScorer(device='cuda:0', lang=lang)

src = 'Love you forever.'
hyp = 'Liebe dich für immer.'
ref = 'Ich werde dich immer lieben.'

src_lang, tgt_lang = lang.split('-')
if tgt_lang == 'en':
    recall = np.array(bart_scorer.score([hyp], [ref], batch_size=4)) # generation scores from the first list of texts to the second list of texts.
    precision = np.array(bart_scorer.score([ref], [hyp], batch_size=4)) # generation scores from the first list of texts to the second list of texts.
    F1 = (2 * np.multiply(precision, recall) / (precision + recall))
    # print(recall)  # [-2.511232614517212]
    # print(precision)
    print(F1)
else:
    faithfulness = bart_scorer.score([src], [hyp], batch_size=4)
    print(faithfulness)


# python3 mrt_scripts/metrics_test/bartscore_test/bartscore_test.py
