from src.config.configuration import ConfigurationManager
from src.components.models.transformer import Transformer
from bpetokenizer import BPETokenizer
import torch

eng_tokenizer = BPETokenizer()
eng_tokenizer.load(
    r"artifacts\\tokenization_trainer\\tokenizer\\eng.json",
    mode="json"
)

hin_tokenizer = BPETokenizer()
hin_tokenizer.load(
    r"artifacts\\tokenization_trainer\\tokenizer\\hin.json",
    mode="json"
)
hin_vocab_size=len(hin_tokenizer.vocab)

def create_decoder_self_attention_mask(tokens, pad_id=0):
    """
    tokens: [1, T]
    return: [1, 1, T, T]
    """
    B, T = tokens.shape

    pad_mask = (tokens != pad_id).unsqueeze(1).unsqueeze(2)  # [1,1,1,T]
    causal_mask = torch.tril(torch.ones(T, T, device=tokens.device))

    return pad_mask * causal_mask



def greedy_decode(model, enc_tokens, max_len, device):
    model.eval()

    start_id = hin_tokenizer.special_tokens["<start>"]
    end_id   = hin_tokenizer.special_tokens["<end>"]

    ys = torch.tensor([[start_id]], device=device)

    for _ in range(max_len):
        dec_mask = create_decoder_self_attention_mask(ys)

        with torch.no_grad():
            logits = model(
                enc_in=enc_tokens,
                dec_in=ys,
                decoder_self_attention_mask=dec_mask,
                decoder_cross_attention_mask=None
            )

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_id:
            break

    return ys


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Transformer(
            emb_dim=self.config.emb_dim,
            ffn_hidden=self.config.ffn_hidden,
            num_heads=self.config.num_heads,
            drop_prob=self.config.drop_prob,
            num_layers=self.config.num_layers,
            max_sequence_length=self.config.max_sequence_length,
            hin_vocab_size=hin_vocab_size
        ).to(self.device)

    def translate_sentence(self, sentence, max_len=100):
        self.model.eval()

        # Tokenize English input
        enc_ids = eng_tokenizer.encode(sentence)
        enc_tokens = torch.tensor([enc_ids], device=self.device)

        # Decode
        pred_ids = greedy_decode(self.model, enc_tokens, max_len, self.device)

        # Convert to text
        pred_tokens = pred_ids.squeeze().tolist()

        pred_tokens = [
            i for i in pred_tokens
            if i not in (
                hin_tokenizer.special_tokens["<start>"],
                hin_tokenizer.special_tokens["<end>"],
                hin_tokenizer.special_tokens["<pad>"]
            )
        ]
        pred_text = hin_tokenizer.decode(pred_tokens)

        # Clean output
        pred_text = pred_text.replace("<start>", "").replace("<end>", "").strip()

        return pred_text


        
        