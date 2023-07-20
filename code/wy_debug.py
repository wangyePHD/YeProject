from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel



text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# print(text_encoder)

embedding = text_encoder.text_model.embeddings
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

text = "this is a car"

token_id = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=tokenizer.model_max_length,
    return_tensors="pt"
).input_ids[0]

print(token_id,token_id.size())

embeds = embedding(token_id)
print(embeds,embeds.size())

