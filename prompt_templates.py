ANSWER_GENERATION_PROMPT_WITHOUT_CONTEXT = """Question: {question}

Answer the user's question based on your understanding. NEVER add explanations, prefixes, or extra characters. If unsure, output "unknown"."""


CONTEXT_JUDGMENT_PROMPT = """As an assistant, you are strictly required to analyze the provided context and decide whether it contains sufficient information to answer the original question.

I will provide you with the original question and optional prior follow-up sub-questions and their intermediate answers. The provided context will be embraced by <context> and </context> tags. 

Your response must follow EXACTLY one of these two formats:
- If the context contains sufficient information: "Yes, the answer for the original question is [answer]"
- If the context is insufficient: "No, I want to ask [subquestion]"

Important rules:
1. When sufficient, [answer] must be the EXACT text snippet from context that answers the question.
2. When insufficient, [subquestion] must be:
   - A SINGLE atomic question
   - Focused on ONE missing fact needed to answer original question
   - Clear and answerable with ONE document retrieval
3. NEVER add ANY explanations, prefixes, or extra characters. Output ONLY the exact required format.

Now, the context and original question are as follows.

<context>{context}</context>

<original_question>{original_question}</original_question>"""


SUB_QUESTION_ANSWER_PROMPT = """As an assistant, you are required to answer the sub-question based on the provided context.

The context will be embraced by <context> and </context> tags. You should evaluate based on the given context.

Your should respond with exactly: "The answer is [answer]"

Important rules:
1. If the context doesn't contain the information, respond with: "No relevant info, need to optimize sub-question".
2. NEVER add ANY explanations, prefixes, suffixes, or extra characters. Output ONLY the exact required format.

Now, the context and sub-question are as follows.

<context>{context}</context>

<sub_question>{sub_question}</sub_question>"""

FORCE_ANSWER_PROMPT = """As an assistant, you are strictly required to provide the FINAL answer to the original question.

The provided context will be embraced by <context> and </context> tags.

Your response must follow EXACTLY this format: "The answer is [answer]"

Important rules:
1. [answer] must be a CONCISE factual phrase. NEVER add explanations.
2. If you do not know the answer, make your best reference.
3. NEVER add ANY prefixes, suffixes, or extra characters beyond the required format.

Now, the context and original question are as follows.

<context>{context}</context>

<original_question>{original_question}</original_question>"""



# 添加更多prompt模板
PROMPT_TEMPLATES = {
    "answer_without_context": ANSWER_GENERATION_PROMPT_WITHOUT_CONTEXT,
    "context_judge": CONTEXT_JUDGMENT_PROMPT,
    "sub_q_answer": SUB_QUESTION_ANSWER_PROMPT,
    "force_answer": FORCE_ANSWER_PROMPT,
}