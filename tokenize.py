import json
from collections import defaultdict

class Tokenizer:
    def __init(self):
        pass

    def get_list_of_unique_words(self, text):
        list_of_words = text.split(" ")
        list_of_unique_words = list(set(list_of_words))
        # print(f"Total words: {len(list_of_words)}\nTotal unique words: {len(list_of_unique_words)}")
        return list_of_unique_words

    def split_all_words_characterwise(self, list_of_unique_words):
        tokenized_words_dict = dict()
        for word in list_of_unique_words:
            if word not in tokenized_words_dict:
                tokenized_words_dict[word] = list(word)
                tokenized_words_dict[word].append('</w>')
        return tokenized_words_dict

    def get_token_count_and_appr(self, tokenized_words_dict):
        token_count_dict = dict()
        token_appr_dict = defaultdict(list)
        for key, value in tokenized_words_dict.items():
            for i in range(len(value)-1):
                if (value[i], value[i+1]) not in token_count_dict:
                    token_count_dict[(value[i], value[i+1])] = 1
                    token_appr_dict[(value[i], value[i+1])] = [{key: (i, i+1)}]
                else:
                    token_count_dict[(value[i], value[i+1])] += 1
                    token_appr_dict[(value[i], value[i+1])].append({key: (i, i+1)})
        return token_count_dict, token_appr_dict

    def get_token_pair_count_and_merge(self, token_count_dict):
        sorted_count_dict = dict(sorted(token_count_dict.items(), key=lambda item: item[1], reverse=True))
        token_with_most_appr = list(sorted_count_dict.keys())[0]
        merged_token = token_with_most_appr[0] + token_with_most_appr[1]
        return token_with_most_appr, merged_token, (token_with_most_appr[0], token_with_most_appr[1])

    def update_tokenization_dict(self, tokenized_words_dict, token_appr_dict, token_with_most_appr, merged_token):
        words_seen_track = []
        for word in token_appr_dict[token_with_most_appr]:
            for k, v in word.items():
                if k in words_seen_track:
                    no_of_occurrence = words_seen_track.count(k)
                    tokenized_words_dict[k][v[0]-no_of_occurrence] = merged_token
                    del tokenized_words_dict[k][v[1]-no_of_occurrence]
                else:
                    tokenized_words_dict[k][v[0]] = merged_token
                    del tokenized_words_dict[k][v[1]]
                words_seen_track.append(k)
        return tokenized_words_dict

    def train_tokenizer_n_steps(self, text, n):
        list_of_unique_words = self.get_list_of_unique_words(text)
        tokenized_words_dict = self.split_all_words_characterwise(list_of_unique_words)
        tokens_merged_track = []
        for i in range(n):
            token_count_dict, token_appr_dict = self.get_token_count_and_appr(tokenized_words_dict)
            token_with_most_appr, merged_token, (tok1, tok2) = self.get_token_pair_count_and_merge(token_count_dict)
            tokenized_words_dict = self.update_tokenization_dict(tokenized_words_dict, token_appr_dict, token_with_most_appr, merged_token)
            tokens_merged_track.append((tok1, tok2))
        return tokenized_words_dict, tokens_merged_track

    def get_token_ids_and_vocab(self, text, n):
        tokenized_words_dict, tokens_merged_track = self.train_tokenizer_n_steps(text, n)
        final_list_of_tokens = []
        vocab = dict()
        for w, tkn in tokenized_words_dict.items():
            for t in tkn:
                final_list_of_tokens.append(t)
        final_list_of_tokens = list(set(final_list_of_tokens))
        for idx, t in enumerate(final_list_of_tokens):
            vocab[t] = idx
        # print(f"Vocab size: {len(final_list_of_tokens)}")
        return vocab, tokens_merged_track

    def save_vocab(self, text, n):
        vocab, tokens_merged_track = self.get_token_ids_and_vocab(text, n)
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        with open("merge.txt", "w") as m:
            for merge in tokens_merged_track:
                m.write(merge[0] + "<m>" + merge[1] + "\n")

    def encode(self, text, vocab_path, merge_path):
        final_merge_list = []
        with open(vocab_path, "r") as v:
            vocab = json.load(v)
        with open(merge_path, "r") as f:
            merge_order = f.read()
        merge_order_list = merge_order.split()
        for merge in merge_order_list:
            final_merge_list.append(tuple(merge.split("<m>")))
        for a, b in final_merge_list:
            print(a, b)
        words = text.split()
        all_tokens = []

        for word in words:
            tokens = list(word) + ['</w>']

            for a, b in final_merge_list:
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == a and tokens[i+1] == b:
                        tokens[i:i+2] = [a + b]
                    else:
                        i += 1

            all_tokens.extend(tokens)
        return [vocab[t] for t in all_tokens]
