This project will evalute a model, if it can understand different variation of romanized nepali. 
first there is a english sentnece and translation of to romanized nepali(baseline) and different variation.
the model is asked to convert the variation to english 
the english translation is then compared to the main sentence using bge-m3.
bge-m3 converts the sentence into vector. Both the reference and the model’s translation get placed on the map. The script measures the distance between the two points (cosine similarity). Close means the translation preserved the meaning, which is the score you see.

I am going home” and “I’m heading back to my house” land near each other.
“What is the weather today?” lands far away
The main problem: embeddings measure topic, not correctness

Embedding similarity answers “are these two texts about the same thing?”, not “does this translation say the same thing?” In machine translation evaluation, that gap matters. Embeddings tend to be weak on:

Pronouns and person: “I am going home” vs. “You are going home” will likely score around 0.9. In Nepali, ma (I) vs. timi (you) is exactly the kind of thing a weak model gets wrong.
Negation: “I am not going home” scores close to “I am going home”.
Tense, numbers, names, and quantities.
Score compression: most English sentence pairs land between 0.5 and 0.95, so differences in the top range are hard to interpret, and there’s no principled “pass” threshold

As human evaluation is the gold standard.
Primary metric: human adequacy, scored simply. Use a 3-level rubric: 2 = same meaning, 1 = partially right , 0 = wrong. Report % fully correct per model per variation type. It’s simple, interpretable, and the standard in small MT studies