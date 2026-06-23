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

SUFFICIENCY_CHECK_PROMPT = """You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide a complete and temporally sufficient answer to the user's query.
--------------------------
User Query:
{query}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Instructions:

1. **Analyze the Query Structure**  
   - Identify key entities AND determine if the query requires temporal reasoning.
   - If the query involves time (e.g., "before", "after", "since", "during", "from X to Y", "how long"), you MUST decompose it into:
       * start_time_needed (if any)
       * end_time_needed (if any)
       * temporal_relation_needed (ordering, duration, interval)

2. **Scan Documents for Coverage**  
   - Look for explicit facts addressing *each* required component:
       * required entities  
       * start time  
       * end time  
       * temporal relations (ordering or duration)

3. **Extract Key Information**  
   - List specific resolved entities or facts found in the documents.
   - If time expressions exist, normalize them (e.g., "two weeks ago", "before she moved").

4. **Identify Missing Information**  
   - For temporal queries:  
        * missing start time  
        * missing end time  
        * missing ordering facts  
        * missing duration  
   - Use resolved names to be specific (e.g., "Start time of Alice moving", "Whether Bob visited before Alice moved").

5. **Judgment**  
   - **Sufficient**: All required components (entities + temporal boundaries + relations) appear explicitly.  
   - **Insufficient**: ANY required part is missing.

### Output Format (strict JSON):
{{
  "is_sufficient": true or false,
  "reasoning": "1-2 sentence explanation.",
  "key_information_found": ["List of resolved entities/facts"],
  "missing_information": ["Specific missing components, using resolved entity names"]
}}

Now evaluate:"""

MULTI_QUERY_GENERATION_PROMPT = """You are an expert at query reformulation for long-term conversational retrieval.
Your goal is to generate multiple complementary search queries that recover BOTH:
- the starting point of a time interval
- the ending point of a time interval
- all temporally-linked events in between

You MUST explicitly expand temporal references (e.g., "last week", "before moving", 
"when they first met") into alternative expressions.

--------------------------
Original Query:
{original_query}

Key Information Found:
{key_info}

Missing Information:
{missing_info}

Retrieved Documents:
{retrieved_docs}
--------------------------

### Temporal Reasoning Strategy (MANDATORY)
When the question involves time or order:
1. **Boundary Decomposition**  
   Generate queries that separately target:
   - the earliest relevant event ("start boundary")
   - the latest relevant event ("end boundary")

2. **Temporal Expression Expansion**  
   Rewrite relative time expressions into multiple equivalent forms:
   - absolute dates (if deducible)
   - session numbers
   - “before/after X”
   - duration phrasing (“two weeks earlier”, “shortly after”)

3. **Interval Reconstruction**  
   Include a declarative query that resembles a hypothetical answer containing BOTH
   the start and end time anchors.

### Standard Query Requirements
1. Generate 2-3 diverse queries.
2. Query 1 MUST be a specific **Question**.
3. Query 2 MUST be a **Declarative Statement or Hypothetical Answer (HyDE)**.
4. Query diversity MUST include different temporal forms (before/after/during).
5. MUST use Key Info to resolve pronouns IF provided.
6. No invented facts.  
7. Keep queries < 25 words, same language as original.

### Output Format (STRICT JSON):
{{
  "queries": [
    "Refined query 1",
    "Refined query 2",
    "Refined query 3 (optional)"
  ],
  "reasoning": "Brief explanation of how temporal boundaries and express.
}}"""

PROMPT_TEMPLATES = {
    "answer_without_context": ANSWER_GENERATION_PROMPT_WITHOUT_CONTEXT,
    "context_judge": CONTEXT_JUDGMENT_PROMPT,
    "sub_q_answer": SUB_QUESTION_ANSWER_PROMPT,
    "force_answer": FORCE_ANSWER_PROMPT,
    "sufficient_check": SUFFICIENCY_CHECK_PROMPT,
    "sub_q_generate": MULTI_QUERY_GENERATION_PROMPT,
}