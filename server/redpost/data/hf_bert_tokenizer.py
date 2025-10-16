import re
from transformers import BertTokenizer


# HuggingFace BERT Tokenizer Wrapper
class HfBertTokenizer:
    def __init__(self, huggingface_tokenizer_dir):
        self.tokenizer = BertTokenizer.from_pretrained(huggingface_tokenizer_dir)

    def tokenize(self, text, recover_unk=False):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if recover_unk:
            try:
                tokens = self._recover_unk(text.lower(), tokens)
            except Exception as e:
                print(e)
                pass
        return tokens, tokens_id

    def _recover_unk(self, text, tokens):
        has_unk = False
        for tok in tokens:
            if tok == "[UNK]":
                has_unk = True
                break
        if not has_unk:
            return tokens

        new_tokens = []

        # Fast recover START
        if re.match(r"^[^a-zA-Z0-9']+$", text):
            tmp_text = text.replace(" ", "")
            if len(tmp_text) == len(tokens):
                print("fask recover start")
                success = True
                for t, tok in zip(tmp_text, tokens):
                    if tok == "[UNK]":
                        pass
                    else:
                        if t != tok:
                            success = False
                            break
                    new_tokens.append(t)
                if success:
                    print(f"fask recover success: '{tokens}' --> '{new_tokens}'")
                    return new_tokens
        # Fast recover END

        pre_text = ""
        recovered = False
        for i, token in enumerate(tokens):
            if token == "[UNK]":
                print("before recover:", i, tokens, flush=True)
                if i+1 < len(tokens):
                    post_token = tokens[i+1].replace("##", "")
                else:
                    post_token = ""
                pattern = f"{pre_text}(.*?){post_token}(.*?)"
                print(f"pattern ({pattern})", flush=True)
                try:
                    token = re.match(pattern, text).group(1)
                except Exception as e:
                    print(f"!!! {i} text=({text}) # ({tokens[i-3:i+3]}) # pattern=({pattern}) # {e}", flush=True)
                    raise ValueError
                print("recover [UNK] ->", token)
                new_tokens.append(token)
                pre_text += token
                recovered = True
            else:
                new_tokens.append(token)
                pre_text += token.replace("##", "")
        if recovered:
            print("after recover", new_tokens)
        return new_tokens

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        raise NotImplementedError

