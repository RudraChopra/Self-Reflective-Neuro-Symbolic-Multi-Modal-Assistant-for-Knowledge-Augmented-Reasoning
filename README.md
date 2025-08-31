Goal, a compact multimodal assistant for knowledge augmented VQA that retrieves wiki facts only when needed, critiques its own answer, applies simple rules, and returns the answer with evidence.

Today plan,
one, run a ready VQA baseline on a few samples
two, add a very small wiki retrieval and citation
three, add one reflection check and one rule check


## Latest baseline results

BLIP full pipeline with retrieval and reflection  
accuracy about 0.71 on 7 live items  
evidence coverage about 0.57  
mean latency about 9.06 seconds  
p95 latency about 10.80 seconds

Ablations  
BLIP rules off accuracy about 0.29  
BLIP VQA only accuracy about 0.29

Figures live in docs slash figs  
Error examples live in docs slash error_examples.md
