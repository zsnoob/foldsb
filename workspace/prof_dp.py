import os
import vllm
from vllm import LLM
from vllm.utils import get_open_port
from vllm import LLM, SamplingParams


GPUs_per_dp_rank = 1
DP_size = 2


def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                              GPUs_per_dp_rank))

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.8,
                                    top_p=0.95,
                                    max_tokens=16 * (dp_rank + 1))

    llm = LLM(model="../scripts/DeepSeek-V3",
            trust_remote_code=True, tensor_parallel_size=GPUs_per_dp_rank, 
            enable_expert_parallel=True, enforce_eager=True)

    outputs = llm.generate(input_text, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    # with open('4k_token.txt', 'r', encoding='utf-8') as file:
    #     input_text = file.read()
    # outputs = llm.generate(input_text)

    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    from multiprocessing import Process
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
