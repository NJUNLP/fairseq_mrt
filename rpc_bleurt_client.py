import sys
import time
import xmlrpc.client


proxy = xmlrpc.client.ServerProxy('http://localhost:8888')

s = time.time()
for i in range(1):
    scores = proxy.bleurt_score((['123', '456'],['123', '789']), int(sys.argv[1]))
    print(scores)
#print(time.time() - s)

# python3 rpc_bleurt_client.py 0