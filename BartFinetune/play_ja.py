from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

print("Start")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
print("model loaded")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
print("loaded")

tmp_text = "Can't believe that I haven't figure out by now"

model_inputs = tokenizer(tmp_text,return_tensors="pt")
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id = tokenizer.lang_code_to_id["ja_XX"]
)
print("generated")
translated = tokenizer.batch_decode(generated_tokens)
print(f"{translated=}")