import codecs

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_path = r"D:\code\nlp20\Week07\output_Qwen1.5"

def bio2word_tags(lines, tags):
    pairs = []
    cur_word = ''
    cur_tag = ''
    for char, tag in zip(lines, tags):
        if tag.startswith('B-'):
            cur_word = char
            cur_tag = tag[2:]
        elif tag.startswith('I-'):
            cur_word += char
        else:
            if cur_word:
                pairs.append((cur_word + ' : ' + cur_tag))
            cur_word = ''
            cur_tag = ''
    if cur_word:
        pairs.append((cur_word + ' : ' + cur_tag))
    if len(pairs) == 0:
        pairs.append('没有识别出待选的实体')

    return pairs

device = 'cuda'

train_lines = codecs.open('../Week07/msra/train/sentences.txt', encoding='utf-8').readlines()[:1000]
train_lines = [line.replace(' ', '').strip() for line in train_lines]

train_tags = codecs.open('../Week07/msra/train/tags.txt', encoding='utf-8').readlines()[:1000]
train_tags = [line.strip().split(' ') for line in train_tags]

train_data = []

for lines, tags in zip(train_lines, train_tags):
    train_data.append([''.join(lines), '\n'.join(bio2word_tags(lines, tags))])
# print(train_data)

val_lines = codecs.open('../Week07/msra/val/sentences.txt', encoding='utf-8').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('../Week07/msra/val/tags.txt', encoding='utf-8').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]

val_data = []
for lines, tags in zip(val_lines, val_tags):
    val_data.append(["".join(lines), "\n".join(bio2word_tags(lines, tags))])

def init_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    return tokenizer, model

def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n现在进行实体识别任务<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断超过最大长度的序列
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def setup_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model

def setup_training_args():
    return TrainingArguments(
        output_dir='./output_ner_Qwen3',
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )

def predict_intent(model, tokenizer, text, device='cpu'):
    messages = [
        {'role': 'system', 'content': '现在进行实体识别任务'},
        {'role': 'user', 'content': text},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([formatted_text], return_tensors='pt').to(device)

    with torch.no_grad():
        generate_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generate_ids = generate_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    return response.strip().split('#')[0]

def batch_predict(model, tokenizer, test_texts, device='cuda'):
    pred_labels = []
    for text in tqdm(test_texts, desc='预测意图'):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append('')
    return pred_labels

def main():
    global train_data
    train_data = pd.DataFrame(train_data)
    train_data.columns = ['instruction', 'output']
    train_data['input'] = ''
    train_data.columns = ['instruction', 'output', 'input']
    ds = Dataset.from_pandas(train_data)

    tokenizer, model = init_model_and_tokenizer(model_path)

    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])
    eval_ds = Dataset.from_pandas(ds.to_pandas().iloc[-200:0])

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

    model.enable_input_require_grads()
    model = setup_lora(model)

    training_args = setup_training_args()

    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()
    trainer.save_model()  # 保存 LoRA adapter 到 output_dir
    tokenizer.save_pretrained("./output_ner_Qwen3/")

def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    tokenizer, model = init_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter(r'D:\code\nlp20\Week07\output_ner_Qwen3', adapter_name="ner_adapter") # 保存的模型路基
    model.cpu()

    # 测试预测
    test_text = "帮我导航到北京的百度大厦"
    result = predict_intent(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"{result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_single_example()