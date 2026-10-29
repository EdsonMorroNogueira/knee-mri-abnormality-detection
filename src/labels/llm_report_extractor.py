"""Extract the 12 knee-MRI classes from a radiology report with a local LLM.

Replaces the earlier regex extractors: the model reads the whole report and
answers all 12 classes at once as JSON. On Kaggle, run with a GPU and
"Internet on" (only needed to download the model). (Stage 2.)

Note: the prompt instructions were translated from Portuguese to English for
this repo, so the prompt differs from the one used in the validated runs.
Report keywords inside it (tear, torn, effusion, ...) are matching data.
"""

import json
import re

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# Short per-class definitions reusing the vocabulary validated in the earlier
# regex extractors, so the small model gets concrete cues. Descriptions were
# translated to English (changes the prompt vs. what was validated); the
# report keywords after "terms:" are unchanged.
CLASS_DEFINITIONS = {
    "ACL": "rupture/tear of the anterior cruciate ligament - terms: tear, torn, rupture, disruption, deficient.",
    "MCL": "injury of the medial collateral ligament - terms: sprain, tear, torn, injury.",
    "Medial Meniscus": "injury/tear of the medial meniscus - terms: tear, " "torn, degeneration, maceration.",
    "Lateral Meniscus": "injury/tear of the lateral meniscus - terms: tear, " "torn, degeneration, maceration.",
    "Medial OA": "osteoarthritis of the medial compartment "
    "of the knee - terms: osteoarthritis, degenerative "
    "changes, joint space narrowing.",
    "Lateral OA": "osteoarthritis of the lateral compartment "
    "of the knee - same terms as Medial OA, but "
    "on the lateral side.",
    "PF OA": "patellofemoral osteoarthritis - terms: "
    "patellofemoral osteoarthritis, patellofemoral joint "
    "degenerative changes.",
    "Effusion": "joint effusion / fluid in the joint - terms: " "joint effusion, effusion.",
    "Synovitis": "inflammation of the synovial membrane - terms: synovitis, " "synovial thickening, synovial hypertrophy.",
    "Baker's": "Baker's cyst (popliteal cyst) behind the " "knee.",
    "Contusion": "bone contusion (bone bruise) - bone marrow edema " "from trauma, without fracture.",
    "Fracture": "bone fracture - terms: fracture, fractured.",
}


def build_prompt(report_text: str) -> str:
    """Build the prompt asking for all 12 classes at once, as JSON.

    The model must answer with exactly one JSON object with the 12
    TARGET_COLUMNS keys, each 0 or 1 (1 = finding present and affirmed in
    the report, 0 = absent or negated). The full report is embedded.
    """
    definitions_text = "\n".join(f"- {col}: {CLASS_DEFINITIONS[col]}" for col in TARGET_COLUMNS)
    columns_json_example = ", ".join(f'"{col}": 0' for col in TARGET_COLUMNS)

    return f"""You are a radiologist reading a knee MRI report. Your task is to \
decide, for each of the 12 findings below, whether it is PRESENT and \
AFFIRMED in the report (1) or ABSENT/NEGATED (0).

Findings to evaluate:
{definitions_text}

Negation rule: if a negation word such as "no", "without", "not", \
"negative for" or "absence of" appears BEFORE the finding's keyword in the \
same sentence, consider the finding ABSENT (0). If the negation appears \
AFTER the finding's keyword, it may be negating something else in the \
sentence - do not count it as a negation of the finding.

Report:
\"\"\"
{report_text}
\"\"\"

Answer with ONLY a JSON object, with no text before or after it, in \
this exact format (replace the values with 0 or 1 for this report):
{{{columns_json_example}}}
"""


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_llm():
    """Load the model and tokenizer. Needs a GPU and, on first run, internet
    to download the model from Hugging Face."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate the model response for the prompt as plain text."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_llm_json(response_text: str) -> dict:
    """Extract the JSON from the model response into a dict with the 12
    TARGET_COLUMNS keys, each 0, 1 or None.

    None means the class could not be extracted (missing or malformed JSON,
    or missing key); it must not raise and abort the whole report.
    """
    # Keep only the span between the first '{' and last '}' (the model may wrap the JSON in text or ```json fences).
    start = response_text.find("{")
    end = response_text.rfind("}")

    json_chunk = response_text[start : end + 1]

    try:
        parsed = json.loads(json_chunk)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        return {col: None for col in TARGET_COLUMNS}

    result = {}
    for col in TARGET_COLUMNS:
        if col in parsed:
            result[col] = int(parsed[col])
        else:
            result[col] = None
    return result


def extract_all_classes_llm(model, tokenizer, report_text: str) -> dict:
    """Chain build_prompt, generate_response and parse_llm_json."""
    prompt = build_prompt(report_text)
    response_text = generate_response(model, tokenizer, prompt)
    return parse_llm_json(response_text)


if __name__ == "__main__":
    example_report = "Impression: 1. Small joint effusion. 2. No fracture identified. " "3. ACL is intact."
    prompt = build_prompt(example_report)
    assert isinstance(prompt, str)
    assert example_report in prompt
    for col in TARGET_COLUMNS:
        assert col in prompt, f"Prompt does not mention class {col!r}"
    print("build_prompt OK - prompt has", len(prompt), "characters")

    parse_examples = [
        (
            '{"ACL": 0, "MCL": 0, "Medial Meniscus": 0, "Lateral Meniscus": 0, '
            '"Medial OA": 0, "Lateral OA": 0, "PF OA": 0, "Effusion": 1, '
            '"Synovitis": 0, "Baker\'s": 0, "Contusion": 0, "Fracture": 0}',
            {"Effusion": 1, "Fracture": 0},
        ),  # clean JSON
        (
            'Aqui está minha análise:\n```json\n{"ACL": 1, "Fracture": 0}\n```',
            {"ACL": 1, "Fracture": 0},
        ),  # JSON inside a code block, with surrounding text
        ("Desculpe, não consigo processar este laudo.", {col: None for col in TARGET_COLUMNS}),  # no JSON at all
    ]
    for response_text, expected_subset in parse_examples:
        parsed = parse_llm_json(response_text)
        assert set(parsed.keys()) == set(TARGET_COLUMNS), f"Missing keys: {set(TARGET_COLUMNS) - set(parsed.keys())}"
        for key, expected_value in expected_subset.items():
            assert parsed[key] == expected_value, f"{key}: expected {expected_value}, got {parsed[key]}"
        print(f"[OK] {response_text[:50]!r}... -> {parsed}")
