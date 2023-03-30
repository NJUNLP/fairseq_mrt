hyps = ['a', 'b', 'c']
refs = ['d', 'e', 'f']
srcs = ['1', '2', '3']

data = []
for h, r, s in zip(hyps, refs, srcs):
    sample = {'src': s, 'mt': h, 'ref': r}
    data.append(sample)

print(data)

# python3 mello_scripts/comet_test/zip_test.py