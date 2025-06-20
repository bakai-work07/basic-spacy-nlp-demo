import spacy

nlp = spacy.load('en_core_web_md')

with open("wiki_us.txt", "r") as file:
    text = file.read()

doc = nlp(text)

# Sentence Segmentation - Splitting text into sentences
print("\nFirst 3 Sentences:\n")
for sent in list(doc.sents)[:3]:
    print(sent.text)

# Identifying POS (Parts Of Speech) using SpaCy
print("\nFirst 50 tokens (Lemma, POS, dependency)\n")
for token in doc[:50]:
    print(f"{token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:6} | Dep: {token.dep_}")

# Named Entity Recognition - classifying entities into categories
print("\nNamed Entities (first 50):\n")
for ent in doc.ents[:50]:
    print(f"{ent.text:50} | Label: {ent.label_}")

# Word Vectors and Similarity
token1 = nlp("military")[0]
token2 = nlp("army")[0]
print(f"\nSimilarity between '{token1.text}' and '{token2.text}': {token1.similarity(token2):.4f}")


# Noise reduction - Cleaning tokens
def clean_tokens(doc):
    cleaned = []
    for token in doc:
        if (
            not token.is_stop and
            not token.is_punct and
            not token.is_space and
            not token.like_num and
            not token.is_currency
        ):
            cleaned.append(token.lemma_.lower())
    return cleaned

cleaned_tokens = clean_tokens(doc)
print("\nFirst 50 Cleaned Tokens:\n")
print(cleaned_tokens[:50])
