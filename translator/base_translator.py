from typing import List

from deep_translator import GoogleTranslator, MyMemoryTranslator
from pandas.core.frame import DataFrame
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")


class Helsinki:
    """Artificial intelligence model for translating from Spanish to English"""

    def __init__(self):
        pass

    def translate_text(self, text: str, target: str = "en") -> str:
        translator = GoogleTranslator(source="auto", target=target)
        translation = translator.translate(text)
        return translation

    def translate_batch(self, texts: List[str]) -> List[str]:
        translated = MyMemoryTranslator("es", "en").translate_batch(texts)
        return translated

    def translate_table(self, table: DataFrame) -> DataFrame:
        self.df_table = table
        data = (self.df_table).copy()
        for i in data.columns:
            values_ = []
            if data[i].dtype == "object":
                res = data[i].unique()
                keys_ = [val for val in res if val is not None]
                try:
                    num_char = int(keys_[0])
                    break
                except:
                    values_ = [self._translate_text_ai(j) for j in keys_]
                    dict_ = dict(zip(keys_, values_))

                    def translate_columns(x):
                        try:
                            return dict_[x]
                        except:
                            return x

                    data[i] = data[i].apply(translate_columns)
        return data

    def _translate_text_ai(self, text: str) -> str:
        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
