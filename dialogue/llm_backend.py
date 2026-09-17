"""Chat backend for the step-5 scripts: vLLM on a full GPU, plain transformers on a MIG slice (vLLM 0.11 crashes on MIG UUIDs)."""
import os, torch


class Chat:
    def __init__(self, model, max_model_len=2048, gpu_mem=0.9, backend=None):
        mig = 'MIG' in os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.backend = backend or ('hf' if mig else 'vllm')
        if self.backend == 'vllm':
            from vllm import LLM
            self.llm = LLM(model=model, dtype='bfloat16', max_model_len=max_model_len, gpu_memory_utilization=gpu_mem)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tk = AutoTokenizer.from_pretrained(model); self.tk.padding_side = 'left'
            self.m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map='cuda').eval()
        print(f'[Chat] backend={self.backend} model={model}', flush=True)

    def __call__(self, msgs_list, temperature=0.0, top_p=1.0, max_tokens=512, seed=0, batch=16):
        if self.backend == 'vllm':
            from vllm import SamplingParams
            sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
            return [o.outputs[0].text for o in self.llm.chat(msgs_list, sp, use_tqdm=True)]
        torch.manual_seed(seed); outs = []
        for i in range(0, len(msgs_list), batch):
            prompts = [self.tk.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list[i:i + batch]]
            enc = self.tk(prompts, return_tensors='pt', padding=True).to('cuda')
            with torch.no_grad():
                g = self.m.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, pad_token_id=self.tk.pad_token_id or self.tk.eos_token_id)
            outs += self.tk.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f'[Chat] {min(i + batch, len(msgs_list))}/{len(msgs_list)}', flush=True)
        return outs
