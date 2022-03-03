# import packages
import time

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# model's path
model_name = "deepset/roberta-base-squad2"

# a) Get predictions

# model
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# input
QA_input = {
    'question': 'Which name is also used to describe the Amazon rainforest in English?',
    'context': "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain 'Amazonas' in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
}

# prediction
# 'start' and 'stop' are variables used to evaluate the time needed to get the answer
###################
start = float(time.time())
###################
res = nlp(QA_input)
###################
stop = float(time.time())
diff_time = round(stop - start, 4)
###################

# print result
print(f"[INFO] result = {res}")
print(f"[INFO] time = {diff_time}")

# b) Load model & tokenizer

# type(model) = <class 'transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering'>
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# type(tokenizer) = <class 'transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast'>
tokenizer = AutoTokenizer.from_pretrained(model_name)

# write structure on text file
with open("model_TRANSFORMERS.txt", 'w') as f:
    f.write(str(model))
    f.close()
with open("tokenizer_TRANSFORMERS.txt", 'w') as f:
    f.write(str(tokenizer))
    f.close()

# NOTE: This is version 2 of the model. If you'd like to use version 1, specify revision="v1.0" when
# loading the model in Transformers 3.5. For exmaple:
#
# nlp = pipeline(model=model_name, tokenizer=model_name, revision="v1.0", task="question-answering")