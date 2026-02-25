"""Batch test all questions with a DashScope OpenAI-compatible model (parallel).

Reads every entry in `question (2).jsonl`, asks the model to return only the
final answer (no reasoning), runs calls in parallel, writes per-question
intermediate outputs in separate folders, then merges to
`answer/answers_qwen3.5-vl-plus.jsonl` (validation-style `question`/`answer`).
Reasoning logs stay per question in `log/qwen3.5-vl-plus_<id>.log`.

The model name can be overridden via env `MODEL_NAME` (default: qwen2.5-vl-72b-instruct).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
QUESTION_PATH = ROOT / "question (2).jsonl"
ANSWER_DIR = ROOT / "answer"
LOG_DIR = ROOT / "log"
ANSWER_PATH = ANSWER_DIR / "answers_qwen3.5-vl-plus.jsonl"
PARTS_DIR = ROOT / "answer_parts"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-plus")

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请直接给出最终答案，不要输出推理过程，"
    "也不要添加额外说明或格式。"
)


def load_questions(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            yield item["id"], item["question"]


def save_answer(file_path: Path, question_text: str, answer: str) -> None:
    payload = {"question": question_text, "answer": answer}
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_log(log_dir: Path, qid: int, question_text: str, reasoning: str | None, answer: str) -> None:
    log_path = log_dir / f"qwen3.5-vl-plus_{qid}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Question ID: {qid}\n")
        f.write(f"Question: {question_text}\n\n")
        f.write("Reasoning:\n")
        f.write((reasoning or "<no reasoning returned>") + "\n\n")
        f.write("Answer:\n")
        f.write(answer + "\n")


def extract_reasoning(message) -> str | None:
    # openai objects allow attribute or dict-style access; handle both defensively.
    if hasattr(message, "reasoning_content"):
        return message.reasoning_content
    if isinstance(message, dict):
        return message.get("reasoning_content")
    return None


def get_api_key() -> str:
    key = "sk-901add2aa1c248d38a4d0432416cf662"
    if not key:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running."
        )
    return key


def worker(qid: int, question_text: str, api_key: str) -> Tuple[int, str, str]:
    # Per-call client to avoid thread-safety surprises.
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=False,
    )

    message = response.choices[0].message
    answer = (message.content or "").strip()
    reasoning = extract_reasoning(message)
    return qid, answer, reasoning or ""


def write_part(part_dir: Path, qid: int, question_text: str, answer: str) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part_{qid}.jsonl"
    payload = {"question": question_text, "answer": answer}
    with part_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def merge_parts(parts_dir: Path, answer_path: Path, order: List[int]) -> None:
    if answer_path.exists():
        answer_path.unlink()

    with answer_path.open("w", encoding="utf-8") as out:
        for qid in order:
            part_path = parts_dir / f"part_{qid}.jsonl"
            if not part_path.exists():
                continue
            with part_path.open("r", encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    continue
                data = json.loads(line)
                payload = {"id": qid, "answer": data["answer"]}
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(max_workers: int = 10) -> None:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale part files to prevent accidental merge of old results.
    for part_file in PARTS_DIR.glob("part_*.jsonl"):
        part_file.unlink(missing_ok=True)

    api_key = get_api_key()

    questions: Dict[int, str] = {qid: q for qid, q in load_questions(QUESTION_PATH)}
    order = sorted(questions.keys())

    future_to_qid = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for qid, question_text in questions.items():
            future = executor.submit(worker, qid, question_text, api_key)
            future_to_qid[future] = qid

        for future in as_completed(future_to_qid):
            try:
                qid, answer, reasoning = future.result()
            except Exception as exc:  # noqa: BLE001
                msg = f"<error: {exc}>"
                qid = future_to_qid.get(future, -1)
                if qid in questions:
                    save_log(LOG_DIR, qid, questions[qid], msg, "")
                print(f"failed question {qid}: {exc}")
                continue

            write_part(PARTS_DIR, qid, questions[qid], answer)
            save_log(LOG_DIR, qid, questions[qid], reasoning, answer)
            print(f"done question {qid}")

    merge_parts(PARTS_DIR, ANSWER_PATH, order)


if __name__ == "__main__":
    run()