from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

explainer = SequenceClassificationExplainer(model, tokenizer)
text = "I really love the service but the delivery was awful."
word_attributions = explainer(text)

print("Predicted sentiment:", explainer.predicted_class_name)
print("Reasoning:")
for word, score in word_attributions:
    print(f"{word}: {score:.2f}")