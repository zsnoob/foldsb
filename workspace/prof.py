from vllm import LLM


llm = LLM(model="/your/path/to/DeepSeek-V3",
        trust_remote_code=True, tensor_parallel_size=4, 
        enable_expert_parallel=True, enforce_eager=True)


with open('4k_token.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()
outputs = llm.generate(input_text)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")