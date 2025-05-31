import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from atlastest import EnhancedATLASBlock, OptimizedATLASMetaController, HybridATLASAttention

#
# 1) Load Qwen2.5-0.5B-Instruct on CPU
#
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="cpu",
    torch_dtype=torch.float32
)

hidden_size = model.config.hidden_size
num_heads   = model.config.num_attention_heads
print(f"Model hidden_size={hidden_size}, num_heads={num_heads}")

#
# 2) Build a mock config that matches Qwen and forces KV-heads = Q-heads
#
class MockConfig:
    hidden_size        = hidden_size
    num_attention_heads= num_heads
    num_key_value_heads= num_heads    # override to avoid reshape errors
    intermediate_size  = model.config.intermediate_size
    rms_norm_eps       = getattr(model.config, "rms_norm_eps", 1e-6)

config = MockConfig()

#
# 3) Patch the ATLAS meta-controller so its Linear gets input_size=2*hidden_size+3
#
class PatchedOptimizedATLASMetaController(OptimizedATLASMetaController):
    def __init__(self, hidden_size, cache_size=64, prediction_horizon=4):
        super().__init__(hidden_size, cache_size, prediction_horizon)
        input_size = hidden_size * 2 + 3
        self.complexity_analyzer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 64),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.03),
            torch.nn.Linear(64, 8)
        )

    def _extract_multiscale_features(self, hidden_states, context_length):
        batch, seq_len, hsz = hidden_states.shape
        mean_feat = hidden_states.mean(dim=1)           # (batch, hidden_size)
        std_feat  = hidden_states.std(dim=1)            # (batch, hidden_size)
        ctx_ratio = torch.tensor([context_length/32768.0],
                                 device=hidden_states.device
                                ).expand(batch,1)      # (batch,1)
        comp     = torch.tensor([std_feat.mean().item()],
                                device=hidden_states.device
                               ).expand(batch,1)       # (batch,1)
        probs    = torch.nn.functional.softmax(mean_feat, dim=-1)
        ent      = -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True)
        # total dim = hidden_size + hidden_size + 1 + 1 + 1
        return torch.cat([mean_feat, std_feat, ctx_ratio, comp, ent], dim=-1)

#
# 4) Patch HybridATLASAttention to skip quantization and handle the KV reshape
#
class PatchedHybridATLASAttention(HybridATLASAttention):
    def __init__(self, config):
        super().__init__(config)
        # force KV-heads == Q-heads
        self.num_key_value_heads = self.num_heads

    def forward(self,
                hidden_states,
                attention_mask=None,
                sparsity_mask=None,
                use_mla=False,
                quantization_level=None):
        bsz, seq_len, _ = hidden_states.shape
        # project to QKV
        qkv = self.qkv_proj(hidden_states)                       # (bsz,seq_len,3*hidden)
        qkv = qkv.view(bsz, seq_len, 3, self.hidden_size)        # (bsz,seq_len,3,hidden)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        out, weights = self._efficient_grouped_attention(
            q, k, v, attention_mask, sparsity_mask
        )
        return out, weights, {"attention_type": "GQA"}

    def _efficient_grouped_attention(self,
                                     query,
                                     key,
                                     value,
                                     attention_mask=None,
                                     sparsity_mask=None):
        bsz, seqlen, _ = query.shape
        # split into heads
        q = query.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1,2)
        k = key.view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1,2)
        v = value.view(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1,2)
        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            am = attention_mask[:,None,None,:].to(scores.dtype)
            scores = scores + (am - 1) * 1e9
        # softmax & context
        probs = torch.nn.functional.softmax(scores, dim=-1)
        ctx   = torch.matmul(probs, v) \
                   .transpose(1,2) \
                   .contiguous() \
                   .view(bsz, seqlen, self.hidden_size)
        return self.o_proj(ctx), probs

#
# 5) Build the ATLAS block with our patches
#
atlas_block = EnhancedATLASBlock(config, layer_idx=0)
atlas_block.meta_controller = PatchedOptimizedATLASMetaController(hidden_size)
atlas_block.attention       = PatchedHybridATLASAttention(config)

#
# 6) Single-shot test function
#
def test_atlas(text="Hello, how are you?"):
    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50
    )
    input_ids     = inputs["input_ids"]
    attention_mask= inputs["attention_mask"]

    # get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]

    print("Hidden states shape:", hidden.shape)

    # run ATLAS block
    atlas_out, perf = atlas_block(
        hidden,
        attention_mask=attention_mask,
        input_ids=input_ids
    )

    print("✅ ATLAS completed.")
    print(f" • sparsity_ratio:     {perf['sparsity_ratio']:.1%}")
    print(f" • security_risk:      {perf['security_risk']:.3f}")
    print(f" • attention_type:     {perf['attention_type']}")
    print(f" • energy_efficiency:  {perf['energy_efficiency']:.1f}%")
    print(f" • computational_savings: {perf['computational_savings']:.1f}%")
    return atlas_out, perf

if __name__ == "__main__":
    # use all 8 cores
    torch.set_num_threads(8)
    test_atlas("Hello, how can ATLAS improve my model's performance?")

