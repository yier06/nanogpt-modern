# generate.py
import torch
import torch.nn.functional as F

# 假设你的模型定义在 model.py 中，类名为 GPT
from nanogpt_modern.models import GPT 

def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    自回归生成文本的核心函数
    
    Args:
        model: 实例化后的 GPT 模型
        prompt_ids: 初始提示词的 token id 序列 (Tensor), shape: (1, seq_len)
        max_new_tokens: 需要生成的新 token 数量
        temperature: 温度系数，控制生成的随机性 (1.0为原始分布，<1.0更确定，>1.0更随机)
        top_k: 仅保留概率最高的 k 个 token 进行采样，防止生成乱码
    """
    # 将模型切换到评估模式，关闭 Dropout 等训练专用层
    model.eval() 
    
    # 1. 复制一份 prompt，避免修改原始数据
    idx = prompt_ids.clone()
    
    # 2. 关闭梯度计算，加速推理并节省显存
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # GPT 通常有最大上下文长度限制，如果序列过长，需要截断
            # 注意：这里假设你的模型内部处理了绝对位置编码或 RoPE
            # 如果你的模型有显式的 max_seq_len 属性，可以加上截断逻辑：
            # if hasattr(model, 'max_seq_len'):
            #     idx_cond = idx[:, -model.max_seq_len:]
            # else:
            idx_cond = idx
            
            # 3. 前向传播，获取 logits
            # 假设你的模型返回的是 logits (shape: [batch, seq_len, vocab_size])
            logits = model(idx_cond)
            
            # 4. 只关注最后一个时间步的预测结果
            # shape: (1, vocab_size)
            logits = logits[:, -1, :] 
            
            # 5. 应用温度缩放 (Temperature Scaling)
            if temperature != 1.0:
                logits = logits / temperature
                
            # 6. 应用 Top-K 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将低于第 k 大值的 logit 设为负无穷
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # 7. 将 logits 转换为概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 8. 从概率分布中采样出下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 9. 将新生成的 token 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)
            
    return idx

if __name__ == "__main__":
    # ========== 1. 初始化模型 ==========
    # ⚠️ 请根据你 model.py 中的实际参数修改这里的配置
    from types import SimpleNamespace
    config = SimpleNamespace(
        vocab_size=50257,
        block_size=1024,
        n_layers=12,        # 注意：GPT 里用的是 n_layers，不是 n_layer
        n_heads=12,         # 注意：不是 n_head
        n_embed_size=768,   # 注意：不是 n_embd
        dropout=0.0,        # GPT 里用到了，dict 里没写
    )
    model = GPT(config)
    
    # 因为还没训练，我们用随机权重来测试管道是否跑通
    # 如果后续有了训练好的权重，可以取消下面这行的注释：
    # state_dict = torch.load("path/to/your/checkpoint.pt")
    # model.load_state_dict(state_dict)
    
    # ========== 2. 准备初始 Prompt ==========
    # 假设 "Hello" 对应的 token id 是 15496 (GPT-2 词表)
    # 这里为了测试，我们随机生成一个长度为 5 的序列
    prompt_ids = torch.randint(0, config.vocab_size, (1, 5)) 
    
    print("Initial prompt shape:", prompt_ids.shape)
    
    # ========== 3. 开始生成 ==========
    generated_ids = generate(
        model, 
        prompt_ids, 
        max_new_tokens=20, 
        temperature=1.0, 
        top_k=40
    )
    
    print("Generated sequence shape:", generated_ids.shape)
    print("Generated token IDs:", generated_ids[0].tolist())