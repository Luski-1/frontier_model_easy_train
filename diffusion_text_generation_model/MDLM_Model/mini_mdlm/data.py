from transformers import AutoTokenizer
from functools import partial
import itertools
import datasets
import yaml
import os


def preprocess_and_tokenize(example, text_column, tokenizer, eos):
    """
    把数据tokenize
    """
    text = example[text_column]      # 根据不同dataset，选择对应的列


    tokenizer.padding_side = 'right'                # 指定右填充
    tokenizer.truncation_side = 'right'             # 指定右截断

    tokens = tokenizer(text,
                add_special_tokens=False,        # 不需要增加特殊token
                return_attention_mask=False,     # 不需要返回attention mask
                return_token_type_ids=False)     # 不需要返回bert类型的token type id
    tokens = {'input_ids':[t + [eos] for t in tokens['input_ids']]} # 增加EOS token id

    return tokens

def packing_token_ids(examples, block_size, bos, eos):
    """
    把数据pack
    """
    # 将这批所有样本的 token 拼接成一个长序列
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    new_block_size = block_size - 2  # 预留 bos + eos 两个位置
    total_length = (total_length // new_block_size) * new_block_size    # 整除，不足1024的末尾片段扔掉

    all_input_ids = []
    all_attention_mask = []
    # 按 block 切分，每个 block 前后加 bos/eos
    for i in range(0, total_length, new_block_size):
        seq = [bos] + concatenated_examples[i:i + new_block_size] + [eos]
        all_input_ids.append(seq)
        all_attention_mask.append([1] * block_size)
    
    # batched=True 模式必须返回字典，每个 value 是对应字段的批量列表
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask
    }


def preprocess_dataset(dataset_path, text_column, bos, eos, tokenizer, block_size, packed_path):
    """
    dataset_path: 数据集的本地目录
    text_column: 数据集的文本列
    bos: tokenizer的bos id
    eos: tokenizer的eos id
    tokenizer: tokenizer
    block_size: 本次MDLM模型的可接受的长度，也是数据集pack的长度
    packing_path: 已经完成处理的数据
    return: tokenize&pack的数据集，有input_ids和attention_mask
    """


    # 1、如果有缓存，直接返回即可
    if os.path.isdir(packed_path):
        print(f"目录 {packed_path} 存在")
        return 
    else:
        print(f"目录 {packed_path} 不存在")

    # 2、读取原始数据
    if not os.path.isdir(dataset_path):
        dataset = datasets.load_dataset(            # 通过huggingface的datasets下载
            'JeanKaddour/minipile',
            cache_dir="/workspace/data/cache_dir"
        )
        dataset.save_to_disk("/worksapce/data/miniPile")
    else:
        dataset = datasets.load_from_disk(dataset_path)

    # 3、tokenize
    partial_preprocess_and_tokenize = partial(preprocess_and_tokenize, text_column=text_column, tokenizer=tokenizer, eos=eos)
    tokenized_dataset = dataset.map(partial_preprocess_and_tokenize, batched=True, num_proc=16, desc='Tokenizing')
    tokenized_dataset = tokenized_dataset.remove_columns('text')

    # 3、pack
    partial_packing_token_ids = partial(packing_token_ids, block_size=block_size, bos=bos, eos=eos)
    packed_dataset = tokenized_dataset.map(
        partial_packing_token_ids,
        batched=True,
        num_proc=16,
        desc='Packing',
        remove_columns=tokenized_dataset.column_names  # 移除原始数据集的旧列
    )

    # 4、保存
    packed_dataset.save_to_disk(packed_path)


if __name__ == "__main__":
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    
    # 2、获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name_or_path"])
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer不存在PAD TOKEN 请更换tokenizer或者增加PAD TOKEN替换代码")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token
    EOS = tokenizer.eos_token_id
    BOS = tokenizer.bos_token_id

    
    preprocess_dataset(dataset_path=config["data"]["dataset_path"],
                       text_column=config["data"]["text_column"],
                       bos=BOS, eos=EOS,
                       tokenizer=tokenizer, 
                       block_size=config["model"]["length"], 
                       packed_path=config["data"]["packed_dataset_path"])
    
    print(f"数据前处理已经完成，保存在{config['data']['packed_dataset_path']}")