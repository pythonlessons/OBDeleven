import yaml
from ragas import evaluate
from datasets import Dataset
from rag_model import RagChain
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

with open('configs.yaml', 'r') as file:
    configs = yaml.safe_load(file)

rag_chain = RagChain(**configs)

# Define questions
questions = [
    "Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'", 
    "Pacituok lietuvių liaudies eilėraštį 'Trys gaideliai'",
    "Pacituok lietuvių liaudies eilėraštį 'Saule vire'",
    "Kokie yra dainos 'Du gaideliai' žodžiai?"
]

# Define ground truths
ground_truths = [
    ["Du gaideliai, du gaideliai baltus žirnius kūlė, dvi vištelės, dvi vištelės į malūną vežė. Ožys malė, ožys malė, ožka pikliavojo, o ši trečia ožkytėlė miltus nusijojo. Musė maišė, musė maišė, uodas vandens nešė, saulė virė, saulė virė, mėnesėlis kepė."],
    ['Not enough information.'],
    ['Not enough information.'],
    ["Du gaideliai, du gaideliai baltus žirnius kūlė, dvi vištelės, dvi vištelės į malūną vežė. Ožys malė, ožys malė, ožka pikliavojo, o ši trečia ožkytėlė miltus nusijojo. Musė maišė, musė maišė, uodas vandens nešė, saulė virė, saulė virė, mėnesėlis kepė."],
]

answers = []
contexts = []

# Inference
for query in questions:
    answers.append(rag_chain.invoke(query))
    contexts.append([docs.page_content for docs in rag_chain.retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Evaluate the results
result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=rag_chain.llm,
    embeddings=rag_chain.embeddings
)

df = result.to_pandas()

# save the results to a csv file
df.to_csv('test_results.csv', index=False)