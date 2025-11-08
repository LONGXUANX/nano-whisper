from nanowhisper.config import Config
from nanowhisper.engine.sequence import Sequence, SequenceStatus
from nanowhisper.engine.scheduler import Scheduler

# test调度器和块空间管理器是否能正确分配和调度


def test_prefill():
    """
    测试prefill阶段调度和块分配
    由于block_size=256, 所以每个seq需要3个block,6个cross_block
    5个seq一共需要15个block,30个cross_block
    0 seq: block 0,1,2; cross_block 3,4,5,6,7,8
    1 seq: block 9,10,11; cross_block 12,13,14,15,16,17
    2 seq: block 18,19,20; cross_block 21,22,23,24,25,26
    3 seq: block 27,28,29; cross_block 30,31,32,33,34,35
    4 seq: block 36,37,38; cross_block 39,40,41,42,43,44
    """
    config_tp = Config(model="model/pretrained_models/whisper-large-v3-turbo")
    config_tp.num_kvcache_blocks = 10000
    scheduler = Scheduler(config_tp)
    for i in range(5):
        seq = Sequence([1] * 513)
        scheduler.add(seq)
    scheduler.schedule()
    print("断点打这里看blocks分配结果")

def test_decode():
    config_tp = Config(model="model/pretrained_models/whisper-large-v3-turbo")
    config_tp.num_kvcache_blocks = 10000
    scheduler = Scheduler(config_tp)
    # 不分配块
    seq = Sequence([1] * 512)
    scheduler.add(seq)
    # 要分配块
    seq = Sequence([1] * 512)
    scheduler.add(seq)
    # 要分配块
    seq = Sequence([1] * 768)
    scheduler.add(seq)
    scheduler.schedule()
    """
    这里打断点：
    seq0: block 0,1; cross_block 2,3,4,5,6,7
    seq1: block 8,9; cross_block 10,11,12,13,14,15
    seq2: block 16,17,18; cross_block 19,20,21,22,23,24
    """
    print("断点打这里看blocks分配结果")
    for ID,seq in enumerate(scheduler.running):
        if ID == 0:
            continue
        seq.append_token(1)
        block_table = seq.block_table
        last_block = scheduler.block_manager.blocks[block_table[-1]]
        last_block.hash = 123456789
    scheduler.schedule()
    """
    这里打断点：
    seq0: block 0,1; cross_block 2,3,4,5,6,7
    seq1: block 8,9,【25】; cross_block 10,11,12,13,14,15
    seq2: block 16,17,18,【26】; cross_block 19,20,21,22,23,24
    """
    print("断点打这里看blocks分配结果")

if __name__ == "__main__":
    test_prefill()
    test_decode()
