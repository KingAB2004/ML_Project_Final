"""Score dialogues with ESC-RANK (Zhao et al. 2024, ESC-Eval; https://github.com/AIFlames/Esc-Eval), the
scorer the baseline paper's Table 6 uses: InternLM2-chat-7B with one LoRA adapter per dimension (English
adapters, `<dim>_en`), 0-4 per dialogue. Prompts and dialogue formatting are copied from their score.py;
the base is loaded in 4-bit to fit the 12 GB card (theirs: fp16).

InternLM2's remote code needs an older transformers than the project's, so this runs with a side package
dir first on the path:  PYTHONPATH=~/escrank_pkgs python scripts/escrank_score.py --set ours
  --dialogues runs/data_quality/dialogues/<set>.jsonl  ->  runs/data_quality/escrank/<set>.jsonl (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ADAPTERS = os.path.expanduser(os.environ.get("ESCRANK_DIR", "~/escrank/ESC-RANK"))
DIMENSIONS = ["fluecy", "diversity", "empathy", "suggection effectiveness", "humanoid", "emotional knowledge",
              "human preference"]                     # their spelling, as their adapters were trained on it
ADAPTER_KEYS = ["fluency", "diversity", "empathic", "suggestion", "human", "tech", "overall"]
OUR_NAMES = ["fluency", "diversity", "empathy", "information", "humanoid", "skillfulness", "overall"]

prompt_EN=["I need to evaluate the fluency of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nThe fluency of the conversation is primarily evaluated from two aspects: the fluency individual responses and the overall logical coherence of multi-turn dialogue. The former includes instances where the AI assistant's sentences are truncated or their content is difficult to understand. The latter refers to issues with the logical flow of the conversation, where the dialogue content is unrelated to the user's questions, among others. The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the fluency of individual sentences and the coherence of multi-turn conversations. In particular, most of the AI ​​assistant’s replies are in Chinese.\n2: There are problems with either the fluency of individual sentences or the coherence of multi-turn conversations. In particular, the AI ​​assistant’s replies contains Chinese words.\n3: There are no apparent issues in two issues.\n4: Both the fluency of individual sentences and the coherence of multi-turn conversations are performing exceptionally well.\n","I need to evaluate the diversity of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nDialogue diversity focuses on two aspects: the diversity of dialogue forms and the diversity of dialogue content. The former concerns whether the entire dialogue employs a variety of structures, sentence patterns, and so on, while the latter focuses on the diversity of dialogue content, including topics, suggestions, and more.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the diversity of dialogue forms and the diversity of dialogue content.\n2: There are problems with either the diversity of dialogue forms or the diversity of dialogue content.\n3: There are no apparent issues in two issues.\n4: Both the diversity of dialogue forms and the diversity of dialogue content are exhibited remarkably well.\n","I need to evaluate the empathy of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe empathy of an AI assistant primarily focuses on two aspects: soothing user emotions and analyzing the underlying logic of the problem. The former concerns whether the AI assistant provides emotional comfort, while the latter pertains to whether the AI assistant assists users in logically analyzing the reasons behind emotional responses or the inherent logic of the problem.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: Some statements that may potentially harm users, it can lead to a negative emotional trajectory for the users.\n1: Lacks the provision of emotional comfort to users and fails to assist users in analyzing issues.\n2: Lacks the provision of emotional comfort to users or fails to assist users in analyzing issues.\n3: There are no apparent issues in two issues.\n4: In the dialogue, AI assistant employing a highly personified approach, resembling that of a friend, to appease user emotions and assist users in problem analysis.\n","I need to evaluate the  suggection effectiveness of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe uggection effectiveness of an AI assistant primarily focuses on average advice effectiveness. There are two main considerations, the number of recommendations and the effectiveness of a single recommendation. The number of suggestions is the total number of suggestions given in each round. Whether the suggestions are effective needs to be judged based on the user's questions.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The AI ​​assistant’s suggestions are invalid, and there are even suggestions that may be potentially harmful to the user.\n1: No suggestions or all suggestions are invalid\n2: There are more than five suggestions, but none of them get to the root of the problem or no more than five suggestions, some of which are effective.\n3: There are more than 5 suggestions, some of them are valid or there are no more than 5 suggestions, all of them are valid.\n4: There are more than 5 suggestions, and all of them are valid.\n","I need to evaluate the humanoid of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe humanoid of an AI assistant primarily focuses on the conversation content of AI assistants is different from that of humans.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: AI assistant has obvious AI tendencies, such as structured replies, or saying 'as a large language model'\n2: There are more than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n3: There are two places in the AI ​​assistant’s reply that indicate it is an AI assistant.\n4: There are less than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n","I need to evaluate emotional knowledge of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe emotional knowledge of AI assistants mainly includes five aspects: \n1. Provide emotional comfort \n2. Provide effective suggestions \n3. Provide companionship, encouragement and appreciation \n4. All users’ questions are answered \n5. On the premise that the user has a mental illness, it is recommended to seek professional psychological consultation or have something outstanding.The total score ranges from 0 to 4, if an item appears, 1 point will be added. And here are the rules detials:\n0: One aspect above appears.\n1: Two aspects above appears.\n2: Three aspects above appears.\n3: Four aspects above appears.\n4: Five aspects above appears.\n","I need to evaluate human preference of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the score according to the following rules:\nThe human preference mainly evaluate the degree of human preference towards the responses generated by an AI assistant. After reading the dialogues, please envision yourself as a stressed individual and score the following rules based on the content of the conversation. The total score ranges from 0 to 4. And here are the rules detials:\n0: I do not like this AI assistant.\n1: I do not have any particular feelings.\n2: It's okay, I'll reconsider using it myself.\n3: I will use it when I am stressed.\n4: I will use it myself and recommend it to friends.\n'"]


def dialogue_text(session: dict) -> str:
    lines = []
    for t in session["turns"]:
        if t["role"] == "user":
            lines.append("ESC-Role：" + t["text"])
        elif t["role"] == "supporter":
            lines.append("AI assistant：" + t["text"])
    return "\n\n".join(lines).replace("AI assistant", "**AI助手**").replace("ESC-Role：", "**用户**")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not os.environ.get("ESCRANK_ENABLE"):
        # skipped on the lab (7 Oct): its InternLM2 code needs an older transformers than Python 3.13 can install
        # from wheels, and the 15 GB RAM machine cannot spare memory to build it. Set ESCRANK_ENABLE=1 to run.
        print(f"{args.set}: ESC-RANK skipped (set ESCRANK_ENABLE=1 to run)")
        return
    src = Path(args.dialogues or f"runs/data_quality/dialogues/{args.set}.jsonl")
    out = Path(f"runs/data_quality/escrank/{args.set}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["session_id"] for l in open(out)} if out.exists() else set()
    sessions = [json.loads(l) for l in open(src)][: args.limit]
    todo = [s for s in sessions if s["session_id"] not in done]
    print(f"{args.set}: {len(todo)} of {len(sessions)} dialogues to score", flush=True)
    if not todo:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    base = "internlm/internlm2-chat-7b"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, quantization_config=quant,
                                                 torch_dtype=torch.float16, device_map={"": 0}).eval()   # all on the GPU or fail: no CPU offload
    model = PeftModel.from_pretrained(model, f"{ADAPTERS}/{ADAPTER_KEYS[0]}_en", adapter_name=ADAPTER_KEYS[0])
    for key in ADAPTER_KEYS[1:]:
        model.load_adapter(f"{ADAPTERS}/{key}_en", adapter_name=key)

    with open(out, "a", encoding="utf-8") as fh:
        for n, s in enumerate(todo, 1):
            row = {"session_id": s["session_id"]}
            dialogue = dialogue_text(s)
            for i, dim in enumerate(DIMENSIONS):
                model.set_adapter(ADAPTER_KEYS[i])
                prompt = (prompt_EN[i] + "Dialogue between user and AI assistant: \n" + dialogue.strip() + "\n"
                          + "Based on the rules, give your " + dim + " score (The number only) to the Dialogue.")
                with torch.no_grad():
                    response, _ = model.chat(tok, prompt, do_sample=False, history=[], max_new_tokens=8)
                row[OUR_NAMES[i]] = next((int(c) for c in response if c in "01234"), None)
                row[f"{OUR_NAMES[i]}_raw"] = response
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"  {args.set}: {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
