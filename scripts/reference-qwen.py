#!/usr/bin/env python3
"""Optional independent acceptance reference. NOT used by the .NET product or worker."""
import argparse,json,pathlib,time
p=argparse.ArgumentParser()
p.add_argument('--model-dir',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'models/qwen2.5-3b-instruct')
p.add_argument('--output',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'artifacts/qwen-reference.json')
p.add_argument('--max-tokens',type=int,default=48)
a=p.parse_args()
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
torch.set_num_threads(8)
print('Independent reference:',torch.__version__,transformers.__version__,flush=True)
tokenizer=AutoTokenizer.from_pretrained(str(a.model_dir),local_files_only=True,trust_remote_code=False)
cases=[]
for text in ['用一句话介绍你自己。','What is 2 + 2? Answer briefly.','Hello 世界🌍\n12345',"I'm testing whitespace.\n\n  Hello!\t",'写一句关于月亮的诗。']:
 ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_dict=False)
 cases.append({'prompt':text,'tokenIds':ids})
print('Tokenizer cases prepared; loading FP32 reference weights',flush=True)
model=AutoModelForCausalLM.from_pretrained(str(a.model_dir),local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager')
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=model.to(device).eval();print('Reference device:',device,flush=True)
ids=cases[0]['tokenIds'];input_ids=torch.tensor([ids],device=device)
started=time.monotonic()
with torch.inference_mode():
 logits=model(input_ids=input_ids,use_cache=False).logits[0,-1].float().cpu().tolist()
 generated=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),do_sample=False,max_new_tokens=a.max_tokens,pad_token_id=151643,eos_token_id=[151643,151645])[0,len(ids):].cpu().tolist()
# Product counts output tokens, excluding termination IDs.
while generated and generated[-1] in (151643,151645):generated.pop()
text=tokenizer.decode(generated,skip_special_tokens=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'model':'Qwen/Qwen2.5-3B-Instruct','dtype':'float32','device':device,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizerCases':cases,'prompt':cases[0]['prompt'],'logits':logits,'generatedTokenIds':generated,'text':text,'elapsedMs':(time.monotonic()-started)*1000},ensure_ascii=False)+'\n')
print(text,flush=True);print('Reference saved:',a.output,flush=True)
