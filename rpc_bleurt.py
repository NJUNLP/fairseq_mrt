import sys
import os
import argparse
import socketserver
from xmlrpc.server import SimpleXMLRPCServer
from multiprocessing import Process, Pipe
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class SimpleXMLRPCServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    pass


def inference(idx, recv, args):   
    # 在这里换指标
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
    import tensorflow as tf
    from bleurt import score
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu,True)
    scorer = score.BleurtScorer(args.model)

    # fw = sys.stdout
    while True:
        try:
            batch = recv.recv()
        except EOFError:
            sys.stderr.write(f"inference {idx} finish!\n")
            return
        hyps, refs = batch[0], batch[1]
        scores = scorer.score(references=refs, candidates=hyps)
        assert scores[-1] != 0, "bleurt rpc server failed"   # 验证提醒，确认bleurt服务已启动 TODO 目前还不能用？？
        recv.send(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BLEURT-20 RPC Server")
    parser.add_argument('--model', '-m', type=str, default='bleurt/BLEURT-20')
    parser.add_argument('--process', '-process', type=int, default=1)
    args = parser.parse_args()
    pipes = []
    processes = []
    for i in range(args.process):
        recv, send = Pipe(duplex=True)
        process = Process(target=inference, args=(i, recv, args))
        processes.append(process)
        pipes.append((recv, send))
    for process in processes:
        process.start()
    for pipe in pipes:
        pipe[0].close()

    def bleurt_score(data, idx):
        pipes[idx][1].send(data)
        scores = pipes[idx][1].recv()
        return scores
    server = SimpleXMLRPCServer(("localhost", 8888))
    server.register_function(bleurt_score, "bleurt_score")
    print('#=========================== rpc_bleurt Starting listening! ===========================#')
    server.serve_forever()

    for pipe in pipes:
        pipe[1].close()

    for process in processes:
        process.join()

