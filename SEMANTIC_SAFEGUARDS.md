# Semantic Safeguards

SentenceMaker now layers several lightweight heuristics on top of the LLM validator so we can keep using the faster `gemma2:9b` model without letting nonsensical sentences slip through. These checks all run in pure Python and reuse information we already compute, so they add almost no extra cost per sentence.

## Conditional sanity checks

- If a sentence contains `si`, we require a clear resolution clause. We look for markers such as `entonces`, `puede`, `podrá`, `ayuda`, etc., or a conjugated verb after the comma.
- Sentences that combine `si` and `pero` must still include a resolution marker. This prevents unresolved *“Si … pero …”* statements.
- Templates with both `si` and `pero` receive a scoring penalty when that pattern has been used repeatedly, reducing recurrence.

## Imperative validation

- Imperative templates now pass through `_is_valid_imperative`, which ensures the sentence starts with a known command form from the classified vocabulary (plus a curated fallback list).
- We also scan the generated sentence for at least one noun/proper noun so that commands reference a concrete object instead of abstract concepts.
- Imperative sentences that fail these checks are rejected before LLM validation, forcing a fresh generation.

## Vocabulary/context filters

- **Fantasy vocab filter:** we reject sentences containing `fantasma`, `vampiro`, `ratón gigante`, etc. These words rarely match household/family templates, so we block them outright.
- **Color adjective guards:** if a sentence uses color adjectives, we require at least one tangible context word (household objects, clothing, food, etc.) and reject any sentence that applies color to abstract nouns like `sentimientos` or `salud`.

## Template variety tracking

- We track connector patterns (`si`, `pero`, `cuando`, `porque`) used by recent templates. When a pattern (especially `si+pero`) repeats too often, we penalize its score so other templates get picked. This discourages repetitive structures without extra LLM calls.

## LLM validation remains the last gate

All of the above checks happen before we call `llm_validator.is_coherent()`. Only sentences that pass the heuristics get validated by the LLM, so we keep the stronger grammatical/semantic check without extra model calls. If a sentence fails either the heuristics or the validator, we simply re-prompt the generation (as before).

## Visibility: rejection counters

- The generator now counts how many attempts were rejected by heuristics vs. the validator. The totals appear in both the live progress messages (e.g., `rejections: 12 (heuristic 7, validator 5)`) and the final summary so you can gauge how strict the guardrails are during a run.

## Prompt construction

Every call to the generation LLM receives a structured prompt built inside `sentence_generator._generate_with_llm`:

- **Mood guidance:** each template sets an instruction such as “Create a declarative statement”, “Create an imperative command…”, or “Create a negative imperative…”. Questions are handled explicitly.
- **Slot vocabulary:** for each POS required by the template, up to 20 candidate words are listed, reminding the model to stay inside the available vocabulary.
- **Global requirements:** the prompt reiterates the max word count, insists on grammatical correctness, and warns against repetitive openings (“DO NOT start with these patterns…”), markdown, or double spaces.
- **Structure hints:** connectors present in the template (`y`, `pero`, `si`, `aunque`, etc.) produce short hints (“Use ‘porque’ to explain a reason”, “Use ‘si’ for conditional”). Longer templates also get “complex sentence” hints.
- **Topic seed:** when enough relevant words exist, we prepend a topic (“…about daily routine…”) so the LLM has context instead of random word salad.

These prompt layers work alongside the heuristics so that we start with a better candidate sentence, reducing the number of retries required to satisfy the downstream checks.
