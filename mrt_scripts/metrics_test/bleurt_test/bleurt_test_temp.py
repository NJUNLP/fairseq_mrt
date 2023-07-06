import sys
import xmlrpc.client
from bleurt import score

proxy = xmlrpc.client.ServerProxy('http://localhost:8888')

hyps = ['I like you']
refs = ['I love you']

scores = proxy.bleurt_score((hyps, refs), int(sys.argv[1]))
print('hypo: ' + hyps[0])
print('ref: ' + refs[0] + '      ' + 'bleurt score: ' + str(scores[0]))

# bleurt_checkpoint = "./bleurt/bleurt-large-512"
# bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
# scores = bleurt_scorer.score(references=refs, candidates=hyps)
# print("=================")
# print(scores)

# python3 mrt_scripts/metrics_test/bleurt_test/bleurt_test_temp.py 0