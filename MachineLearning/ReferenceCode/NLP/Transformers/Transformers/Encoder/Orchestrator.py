import Encoder


def PrepareDataSet():
    # Download the dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    checkpoint = 'distilbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)



if __name__ == "__main__":
    """
    '_' is allowed between number for easier readibility
    vocab_size,max_len,d_k, d_model, n_heads,n_layers,n_classes,dropout_prob
    vocab_size = 20,000
    max_len = 1024
    d_k = 16
    d_model = 64
    n_heads = 4
    n_layers = 2
    n_classes = 5
    dropout_prob = 0.1
    """
    model = Encoder(20_000, 1024, 16, 64, 4, 2, 5, 0.1)

#--------------- Dataset preparation
