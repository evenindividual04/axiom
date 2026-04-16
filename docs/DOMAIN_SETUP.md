# Domain-Specific Setup Guide

This guide shows how to adapt Axiom for different sectors and domains. Each example walks through document preparation, prompt customization, and expected metrics.

## Table of Contents

- [Finance](#finance)
- [Healthcare](#healthcare)
- [Legal](#legal)
- [E-commerce](#e-commerce)
- [Customer Support](#customer-support)
- [Code Generation](#code-generation)
- [General Pattern](#general-pattern)

---

## Finance

**Use case**: Evaluate financial AI agents on SEC filings, earnings calls, risk assessments, portfolio analysis.

### 1. Prepare Documents

Collect financial documents and place in `data/fin-docs/`:

```bash
# Example sources
data/fin-docs/
├── 10-k-2024-apple.txt              # Revenue, risk factors, MD&A
├── 10-q-2024-q1-apple.txt           # Quarterly earnings
├── earnings-call-transcript-q1.txt  # Earnings call Q&A
├── financial-metrics-definitions.md # Reference docs
└── gaap-vs-non-gaap-standards.pdf   # Regulatory guidance
```

Or use the default seed documents:
```bash
uv run python scripts/download_seed_docs.py
```

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a financial analysis assistant. Your task is to answer questions about companies, 
financial statements, and market trends accurately and comprehensively.

Guidelines:
- Support all claims with specific numbers from documents
- Distinguish GAAP vs non-GAAP metrics
- Flag assumptions and limitations
- Do not speculate beyond provided data

Question: {question}

Grounding (SEC filings, earnings calls, etc.):
{context}

Expected answer from financial expert:
{ground_truth}

Your response:
```

Create variants in `prompts/improved_prompt.txt`:

```
You are a financial analysis AI assistant with expertise in:
- SEC filing interpretation (10-K, 10-Q, 8-K)
- GAAP accounting standards
- Risk factor assessment
- Valuation methods

Answer the following financial question precisely:

Question: {question}

Available documents:
{context}

Expert answer for reference:
{ground_truth}

Respond in 2-3 paragraphs, citing specific financial metrics. Flag any ambiguities.
```

### 3. Configure Models

Edit `config.yaml`:

```yaml
dataset:
  target_rows: 150                  # More rows for financial precision
  test_type_mix:
    happy_path: 0.50               # Standard questions
    edge_case: 0.35                # Unusual financials, M&A, spinoffs
    adversarial: 0.15              # Trick questions, misdirection

models:
  - name: gpt-4o                   # Best for financial reasoning
    provider: openai
    enabled: true
  - name: claude-3-5-sonnet-latest # Strong on complex analysis
    provider: anthropic
    enabled: true
  - name: gemini-2.0-flash         # Fast baseline
    provider: gemini
    enabled: true

runtime:
  concurrency: 5                   # Higher; financial APIs are fast
  request_timeout_seconds: 90      # Complex analyses may take longer
  rate_limit_per_minute: 10        # Reasonable for finance queries
```

### 4. Add Finance-Specific Metrics

Edit `evals/metrics.py`:

```python
def financial_accuracy(output: str, context: str, ground_truth: str) -> float:
    """
    Check if financial figures are within acceptable range of ground truth.
    Allows for rounding changes, but flags major discrepancies.
    """
    import re
    
    # Extract numbers from output and ground truth
    output_numbers = re.findall(r'\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%', output)
    truth_numbers = re.findall(r'\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%', ground_truth)
    
    if not truth_numbers:
        return 1.0  # No numbers to check
    
    # Simple check: if key numbers are mentioned, score higher
    matches = sum(1 for t in truth_numbers if any(t in o for o in output_numbers))
    return matches / len(truth_numbers) if truth_numbers else 0.0

def no_hallucinated_tickers(output: str, context: str) -> float:
    """
    Ensure output doesn't reference stock tickers or companies not in context.
    """
    # Extract ticker references (AAPL, MSFT, etc.)
    import re
    tickers = re.findall(r'\b[A-Z]{1,5}\b', output)
    mentioned_tickers = set(tickers)
    
    # Check against context
    context_tickers = set(re.findall(r'\b[A-Z]{1,5}\b', context))
    
    hallucinated = mentioned_tickers - context_tickers
    return 1.0 - (len(hallucinated) / max(1, len(mentioned_tickers)))
```

Integrate in `evals/pipeline.py`:

```python
def evaluate_row(...) -> EvalResult:
    # ... existing metrics
    fin_accuracy = financial_accuracy(output, context, ground_truth)
    no_halluc_tickers = no_hallucinated_tickers(output, context)
    
    return EvalResult(
        # ... existing fields
        financial_accuracy=fin_accuracy,
        hallucinated_tickers_count=no_halluc_tickers,
    )
```

### 5. Run Evaluation

```bash
uv run python main.py run-pipeline --live
```

**Expected results**:
- **Success rate**: 80–95% (financial data is complex; some failures normal)
- **Answer Relevancy**: 0.85–0.95 (models should address the question)
- **Faithful**: 0.75–0.90 (trickier; some hallucination of detail expected)
- **Hallucination**: 0.15–0.30 (finance has many edge cases; moderate hallucination okay)
- **Safety**: 0.95–1.0 (financial advice isn't toxic; safety scores high)

**Dashboard insights**:
- Which model best handles risk factor disclosures?
- Do models confuse GAAP vs non-GAAP metrics?
- Which prompt version improves hallucination rate?

---

## Healthcare

**Use case**: Evaluate clinical decision support AI on medical guidelines, treatment protocols, drug interactions.

### 1. Prepare Documents

```bash
data/fin-docs/
├── clinical-practice-guidelines-cardiology.pdf
├── drug-interaction-database.txt
├── patient-safety-protocols.md
├── adverse-event-reporting.txt
└── medical-terminology-standards.pdf
```

**Warning**: Use only public, non-proprietary clinical data. Never include real patient data.

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a clinical decision support AI. Your role is to assist healthcare professionals 
by providing evidence-based information from medical literature and guidelines.

CRITICAL:
- You are NOT a replacement for a licensed clinician
- Always recommend consulting a healthcare provider
- Base recommendations on provided evidence only
- Flag uncertainty and limitations

Clinical Question:
{question}

Clinical Evidence (guidelines, literature):
{context}

Expected clinical response:
{ground_truth}

Your response (2-3 paragraphs, cite evidence):
```

Create `prompts/improved_prompt.txt`:

```
You are a clinical knowledge assistant trained on evidence-based medicine.

For the following clinical scenario, provide an evidence-based response:

Scenario: {question}

Available Evidence:
{context}

Reference Answer:
{ground_truth}

Guidelines:
1. Cite specific guideline sections or studies
2. Note contraindications and adverse effects
3. Recommend only when supported by provided evidence
4. Flag red flags requiring immediate specialist referral
5. Include disclaimer that this is for informational use only

Response:
```

### 3. Configure Models

```yaml
dataset:
  target_rows: 100
  test_type_mix:
    happy_path: 0.40               # Standard clinical questions
    edge_case: 0.45                # Rare conditions, drug interactions, allergies
    adversarial: 0.15              # Trick questions, misdirection

models:
  - name: gpt-4o
    provider: openai
    enabled: true
  - name: claude-3-5-haiku-latest  # Good safety score for clinical
    provider: anthropic
    enabled: true
```

### 4. Add Healthcare-Specific Metrics

```python
def clinical_evidence_citation(output: str, context: str) -> float:
    """
    Score how well output cites evidence from provided context.
    Healthcare requires accountability; vague answers should score low.
    """
    import re
    
    # Look for citation markers or specific references
    citations = len(re.findall(r'(according to|per|based on|guideline|protocol)', output, re.I))
    sentences = len(re.split(r'[.!?]+', output))
    
    # Expect ~1 citation reference per 2–3 sentences in clinical context
    expected_citations = sentences / 3
    return min(1.0, citations / max(1, expected_citations))

def contraindication_awareness(output: str, context: str) -> float:
    """
    Flag if output mentions treatments without noting contraindications.
    """
    import re
    
    # If any treatment/drug mentioned, check for contraindication awareness
    treatments = re.findall(r'(treatment|medication|therapy|drug|antibiotic)', output, re.I)
    
    contraindications = re.findall(r'(contraindicated|contraindication|avoid|caution|not recommended)', output, re.I)
    
    if not treatments:
        return 1.0  # No treatments mentioned, N/A
    
    # Expect at least one caution/contraindication mention
    return min(1.0, len(contraindications) / max(1, len(treatments)))

def safety_disclaimer(output: str) -> float:
    """
    Check if output includes appropriate clinical disclaimer.
    """
    disclaimers = [
        'consult.*doctor',
        'seek.*medical.*advice',
        'not.*substitute.*professional',
        'healthcare.*provider',
    ]
    
    for pattern in disclaimers:
        if re.search(pattern, output, re.I):
            return 1.0
    
    return 0.0  # Missing disclaimer = major issue
```

### 5. Run Evaluation

```bash
uv run python main.py run-pipeline --live
```

**Expected results**:
- **Success rate**: 70–85% (clinical reasoning is complex)
- **Evidence Citation**: 0.70–0.85 (models should back up claims)
- **Contraindication Awareness**: 0.60–0.80 (some miss edge cases)
- **Safety Disclaimer**: 1.0 (critical; models should include or fail)

---

## Legal

**Use case**: Evaluate contract analysis, case law research, regulatory compliance AI.

### 1. Prepare Documents

```bash
data/fin-docs/
├── corporate-law-handbook.pdf
├── contract-templates.txt
├── case-law-summaries-property-law.md
├── ucc-uniform-commercial-code.txt
└── regulatory-compliance-checklist.pdf
```

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a legal research assistant specializing in contract law and case law analysis.

DISCLAIMER: You provide information, not legal advice. Always recommend consulting a licensed attorney.

Legal Question:
{question}

Relevant Case Law & Statutes:
{context}

Expert Legal Analysis:
{ground_truth}

Your analysis (cite precedent and statute):
```

### 3. Add Legal-Specific Metrics

```python
def precedent_citation_accuracy(output: str, context: str) -> float:
    """
    Did output correctly cite precedents and statutes from context?
    """
    import re
    
    # Extract case names and statute numbers
    cases_in_context = re.findall(r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+\b', context)
    cases_in_output = re.findall(r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+\b', output)
    
    if not cases_in_context:
        return 1.0
    
    # Check if cited cases are in context (no hallucinated cases)
    correct_citations = sum(1 for case in cases_in_output if case in cases_in_context)
    
    if not cases_in_output:
        return 0.5  # Should have cited something
    
    return correct_citations / len(cases_in_output)

def legal_conclusion_supported(output: str, ground_truth: str) -> float:
    """
    Does output's conclusion logically follow from provided law?
    """
    # Simple heuristic: check if key conclusion elements are present
    # This is a proxy for logical consistency
    conclusion_words = re.findall(r'(therefore|thus|conclude|ruling|decision)', output, re.I)
    reasoning = re.findall(r'(because|based on|under|pursuant)', output, re.I)
    
    if not conclusion_words:
        return 0.3  # No clear conclusion
    
    if not reasoning:
        return 0.5  # Conclusion without reasoning
    
    return min(1.0, len(reasoning) / max(1, len(conclusion_words)))
```

---

## E-commerce

**Use case**: Evaluate product recommendations, FAQ chatbots, policy compliance.

### 1. Prepare Documents

```bash
data/fin-docs/
├── product-catalog-electronics.txt
├── product-catalog-apparel.txt
├── shipping-policies.md
├── return-policy.txt
├── company-faq.md
└── product-safety-data-sheets.pdf
```

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a customer service AI for our e-commerce platform. 
Your goal is to provide accurate, helpful information about products and policies.

Customer Question:
{question}

Product & Policy Information:
{context}

Correct Answer:
{ground_truth}

Your response (be concise, friendly, accurate):
```

### 3. Add E-commerce-Specific Metrics

```python
def policy_compliance_awareness(output: str, context: str) -> float:
    """
    Did output mention relevant policies (returns, shipping, warranties)?
    """
    keywords = ['return', 'shipping', 'warranty', 'policy', 'guarantee', 'refund']
    
    mentions = sum(1 for kw in keywords if kw in output.lower())
    
    # Expect some policy mention in e-commerce context
    return min(1.0, mentions / 2.0)

def product_recommendation_specificity(output: str, context: str) -> float:
    """
    If recommending products, are specific product names/SKUs mentioned?
    """
    import re
    
    # Look for product names or SKUs
    sku_pattern = r'SKU:?\s*[\w-]+'
    
    product_refs = len(re.findall(sku_pattern, output))
    
    if 'recommend' not in output.lower():
        return 1.0  # No recommendation needed
    
    if product_refs == 0:
        return 0.2  # Vague recommendation
    
    return min(1.0, product_refs / 3.0)  # Expect 2–3 specific products
```

---

## Customer Support

**Use case**: Evaluate support chatbots on FAQ, troubleshooting guides, escalation decisions.

### 1. Prepare Documents

```bash
data/fin-docs/
├── faq-common-issues.txt
├── troubleshooting-flowchart.md
├── escalation-criteria.txt
└── customer-communication-guidelines.txt
```

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a customer support representative for our company.
Your goal is to resolve customer issues helpfully and efficiently.

Customer Issue:
{question}

Knowledge Base:
{context}

What the customer should hear:
{ground_truth}

Your response (empathetic, clear, actionable):
```

### 3. Add Support-Specific Metrics

```python
def empathy_score(output: str) -> float:
    """
    Does output include empathetic language?
    """
    empathy_words = [
        'understand', 'appreciate', 'sorry', 'frustrat', 'help',
        'glad', 'support', 'grateful', 'thank'
    ]
    
    mentions = sum(1 for word in empathy_words if word in output.lower())
    return min(1.0, mentions / 2.0)

def actionable_next_steps(output: str) -> float:
    """
    Does output end with clear next steps for the customer?
    """
    action_words = ['next', 'step', 'please', 'try', 'follow', 'contact', 'submit']
    
    last_sentence = output.split('.')[-1].lower()
    
    has_action = any(word in last_sentence for word in action_words)
    
    return 1.0 if has_action else 0.3
```

---

## Code Generation

**Use case**: Evaluate code generation AI on algorithmic problems, code review, refactoring suggestions.

### 1. Prepare Documents

```bash
data/fin-docs/
├── coding-style-guide.md
├── design-patterns-reference.txt
├── api-documentation.md
├── security-best-practices.txt
└── performance-optimization-tips.md
```

### 2. Customize Prompts

Edit `prompts/base_prompt.txt`:

```
You are a code generation assistant. Write clean, performant, well-documented code.

Task:
{question}

Context & Guidelines:
{context}

Expected code:
{ground_truth}

Your solution (with comments, error handling, tests):
```

### 3. Add Code-Specific Metrics

```python
def code_correctness(output: str, ground_truth: str) -> float:
    """
    Try to parse and validate generated code syntax.
    """
    import ast
    
    try:
        ast.parse(output)
        has_syntax = 1.0
    except SyntaxError:
        has_syntax = 0.0
    
    # Bonus: check if key logic elements are present
    logic_elements = ['def', 'class', 'for', 'if']
    logic_present = sum(1 for elem in logic_elements if elem in output) / len(logic_elements)
    
    return (has_syntax + logic_present) / 2.0

def test_coverage_awareness(output: str) -> float:
    """
    Does code include tests or mention testing?
    """
    test_keywords = ['test', 'assert', 'unittest', 'pytest', 'def test_']
    
    mentions = sum(1 for kw in test_keywords if kw.lower() in output.lower())
    
    return min(1.0, mentions / 2.0)

def security_awareness(output: str) -> float:
    """
    Does code avoid common security pitfalls?
    """
    import re
    
    # Red flags
    red_flags = [
        r'eval\(',              # eval is unsafe
        r'exec\(',              # exec is unsafe
        r"input\(",             # unvalidated input
        r'os\.system\(',        # command injection risk
        r"=.*'\+.*\+"           # SQL injection pattern
    ]
    
    flag_count = sum(1 for pattern in red_flags if re.search(pattern, output))
    
    # Green flags
    green_flags = [
        'parameterized',
        'sanitize',
        'escape',
        'validate',
        'auth',
        'permission'
    ]
    
    green_count = sum(1 for phrase in green_flags if phrase.lower() in output.lower())
    
    # Score: penalize red flags, reward green flags
    return max(0.0, 1.0 - (flag_count * 0.2) + (green_count * 0.1))
```

---

## General Pattern

All domain setups follow this pattern:

### Step 1: Data Preparation
```
- Collect domain documents
- Place in data/fin-docs/
- Supported formats: .txt, .md, .pdf
```

### Step 2: Prompt Customization
```
- Edit prompts/base_prompt.txt (main question format)
- Create improved/advanced variants
- Tailor to domain expertise requirements
```

### Step 3: Configuration
```
- Enable relevant models in config.yaml
- Set appropriate test mix (happy_path, edge_case, adversarial)
- Tune concurrency, timeout, rate limits
```

### Step 4: Custom Metrics
```
- Add domain-specific evaluation functions to evals/metrics.py
- Integrate into evals/pipeline.py
- Update EvalResult dataclass
```

### Step 5: Run & Iterate
```bash
uv run python main.py run-pipeline --live
uv run python main.py dashboard
```

### Step 6: Interpret Dashboard
```
- Overview tab: Success rate + failure breakdown
- Prompts tab: Which variant improves metrics?
- Provider Health tab: Which models are stable?
- Artifacts tab: Links to full manifest, report, CSV
```

---

## Tips for Any Domain

1. **Start small**: Begin with 50 test cases, then scale to 200+
2. **Mix test types**: 50% happy path, 30% edge case, 20% adversarial
3. **Segment by model**: Compare GPT-4o vs Claude vs local models
4. **Version your prompts**: base → improved → advanced; measure delta
5. **Monitor provider health**: Set alerts on rate limits and failures
6. **Preserve artifacts**: Save manifests for reproducibility and audit trails
7. **Iterate fast**: Tweak prompts, rerun evals, check dashboard (2–5 min cycles)

---

## Next: Run Axiom

Choose your domain, prepare docs, customize prompts, and evaluate:

```bash
cd axiom
bash scripts/bootstrap.sh
uv run python main.py run-pipeline --live
uv run python main.py dashboard
```
