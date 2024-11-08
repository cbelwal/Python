# %% [markdown]
# The site: https://paperswithcode.com/ contains code reference for published papers. The original ref. for this code is from https://github.com/bentrevett/pytorch-seq2seq which was referenced in the paperswithcode.com link.
# 
# The code implements the paper "Sequence to Sequence Learning with Neural Networks", Ilya Sutskever, Oriol Vinyals, Quoc V. Le, available at: https://arxiv.org/abs/1409.3215

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm # to display a progress bar
import evaluate

#import import_ipynb # Give it before import of other ipynb
#from Encoder import Encoder

# %% [markdown]
# Set the seed in all libraries so that startup weights and other paramters are same in every run.

# %%
seed = 36

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# %% [markdown]
# The dataset used here is the English to German translation text. Orig. data set is available here: https://github.com/multi30k/dataset. The HF datasets library has access to this under "bentrevett/multi30k". We will load this from HF.
# 
# This Dataset is already split into  Training, Validation and Test groups like most of the datasets in the HF library. 

# %%
dataset = datasets.load_dataset("bentrevett/multi30k")
dataset # Print the data split

# %% [markdown]
# Assign Train, Test and Validation into variables:
# 

# %%
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

# %% [markdown]
# Following are sample contents: train_data[0]
# 
# {'en': 'Two young, White males are outside near many bushes.',
#  'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}

# %% [markdown]
# For Tokenization we will use spaCy which is a newer library than NLTK and tiktoken. We first need to download the tokenization models for each language which will be en and de in this case.
# 
# Unlike NLTK, there is no way to download the models from code in Spacy, hence the following commands have to be run before loading the tokenizations models.
# 
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
# 
# These models are downloaded in the folder:
# .venv\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.7.1
# 

# %%
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

# Print download path
print(en_nlp._path)

# %% [markdown]
# Let us manually call the Tokenizer for Sample Text

# %%
string = "Top Gun is my favorite movie!"
tokens = en_nlp.tokenizer(string)

[token.text for token in tokens]


# %% [markdown]
# Define the function to be used by the map method of the Datasets objects. Add the Start of sentence (sos) and the End of Sentence (eos) tokens which are passed to the function.

# %%
# These arguments can be passed as a kwargs dict.
def tokenize_for_map(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    # max_length will terminate the string if longer than a specific length
    # this step is reapeated for each token
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    # Return as a dict.
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}


# %% [markdown]
# Set the parameters to pass to tokenize_for_map() as kw_args 

# %%
max_length = 1000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

kwargs = {
    "en_nlp": en_nlp,
    "de_nlp": de_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

# %% [markdown]
# Call tokenize_for_map() with above arguements for each object.

# %%
train_data = train_data.map(tokenize_for_map, fn_kwargs=kwargs)
valid_data = valid_data.map(tokenize_for_map, fn_kwargs=kwargs)
test_data = test_data.map(tokenize_for_map, fn_kwargs=kwargs)

# %% [markdown]
# See a sample of the data after the tokenize_for_map() operations

# %%
train_data[0]

# %% [markdown]
# We will build the vocabulary now which is assigning unique token_ids to each token, which serves as a lookup table mapping numbers to tokens. We also assign the Unknown '<unk>' and Pad '<pad>' token. 
# 
# The special_tokens variables is set to a list that will be passed to the torchtext.vocab.build_vocab_from_iterator()
# 
# The min_freq param specifies that only tokens who appear min_freq times should be considered in the dataset. If any token is less than min_freq times it will be treated a <unk> token.  
# 
# The parameters to create the vocab. are specified first

# %%

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

# %% [markdown]
# Now we will call the functions to build both the 'en' and 'de' vocabularies. vocab should only be built from training data, if some token is present in test/validation but in training, then it should be treated as unknown.

# %%
en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["de_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

# %% [markdown]
# Print a sample of the vocab. Print the 1st 10 tokens in the vocab. The regular tokens are orderded from the most frequenct to least frequent, and the specical tokens are not subject to this. itos() shows the string for a given index while stoi() will give the index for a specific string. 

# %%
# NOTE: In .ipynb only the last line is printed
en_vocab.get_itos()[:10], de_vocab.get_itos()[:10]

# %% [markdown]
# Check some stoi() values

# %%


# %%
en_vocab.get_stoi()["my"]


# %% [markdown]
# Can also use the object as a dict.

# %%
en_vocab["my"]

# %% [markdown]
# Get index of special tokens

# %%
en_vocab[unk_token], en_vocab[pad_token], de_vocab[unk_token], de_vocab[pad_token]

# %% [markdown]
# We can also look up indices of multiple words

# %%
tokens = ["my","name","is","the","man"]
en_vocab.lookup_indices(tokens)

# %% [markdown]
# Get size of the vocabulary in both languages

# %%
len(en_vocab), len(de_vocab)

# %% [markdown]
# The vocab. can be used like a Map and similar operations can be performed on it.

# %% [markdown]
# The special tokens will have the same id and using an Assert we can confirm that.

# %%
assert en_vocab[unk_token] == de_vocab[unk_token]
assert en_vocab[pad_token] == de_vocab[pad_token]

# %% [markdown]
# Set the default index. The default index is returned if some token is not found in the vocab. This is a very important step, as you will get 'key' not found errors.

# %%
en_vocab.set_default_index(en_vocab.get_stoi()[unk_token])
de_vocab.set_default_index(en_vocab.get_stoi()[unk_token])

# %% [markdown]
# The following function will return the indices for any passed group of string and will behave in a similar way like we call map. 

# %%
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids} 

# %% [markdown]
# Using the above function let's add the token ids to the train, validation and test data

# %%
fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

# Add the en_ids and de_ids rows
train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

# %% [markdown]
# Check one row and see the additional column that is added. 'en_ids' and 'de_ids' contain the ids (or token ids) of the words:

# %%
train_data[0]

# %% [markdown]
# Convert specific columns of en_ids and de_ids to torch tensors.We will use the with_format() function.

# %%
data_type = "torch"
format_columns = ["en_ids", "de_ids"]

train_data = train_data.with_format(
    type="torch", columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(type="torch",
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(type="torch",
    columns=format_columns,
    output_all_columns=True,
)

# %% [markdown]
# Torch's  DataLoader class will be used to create the batch. Dataloader can call special functions when creating batches, and we will use a collate (Combine) function to pad the input sequence. Padding of input sequence is important as the matrix for weights is fixed. pad_sequence() of nn.utils will be used to pad the sequence.
# 
# Note that we use a closure type construct for get_collate_fn(). When DataLoader calls the collage function it only sends it the batch, as function param and not the pad_index. By using the function within a function the pad_index value needs to be passed once, and then collate_fn() can be called directly and will use the pad_index value defined before.

# %%

def get_collate_fn(pad_index): # This () is called once, assigns value of pad_index
    def collate_fn(batch):     # Called by the dataloader.
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn

# %% [markdown]
# Create the get_data_loader() that will call the collate(). This will return the DataLoader objects. This function is called multiple times for train, test and validation sets.

# %%
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index) # Get collate_fn as a value, same pad_index will be used
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

# %% [markdown]
# Now call get_data_loader() for train, test and validation. Set to a high batchs size, if GPU is available use the largest batch size that will fit in GPU memory. For training, data should be shuffled but not needed for test and validation.

# %%
batch_size = 1
pad_index = en_vocab[pad_token]

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# %% [markdown]
# At this point we are ready to build the model. Model will be built in 3 parts, the Encoder, Decoder then the seq2seq linkage between the two. Note that this uses the nn.Embedding layer of pytorch, so embeddings weights will also be learned (we are not using any pretrained embeddings like word2vec). 

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        print(f"Input dim {input_dim}, embedding dim {embedding_dim}")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # LSTM(# input size, #hidden, # layers), # hidden also corresponds to output count 
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout) # dropout = prob. of dropout (for randomly zeroing the input tensor values)


    # src is the en_ids, or list of token ids in a sentence
    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # LSTM Input:
        # input: tensor of shape  (L,Hin) for unbatched input,  (L,N,Hin)  when batch_first=False or  
        # (N,L,Hin) when batch_first=True containing the features of the input sequence.
        # L = src length, N = batch size, Hin = embedded dimension
        #
        # embedded = [src length, batch size, embedding dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # LSTM Output format:  output, (h_n, c_n)
        # output: Output features for each t 
        # h_n: final hidden state for each LSTM Cell
        # c_n: final cell state for each LSTM Cell
        #
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top/last hidden layer
        return hidden, cell

# %% [markdown]
# The code for Decoder is shown next. This has a FC layer in the end to allow for a softmax like output to predict the probabilites of each token.

# %%
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim) # 1
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout) #2
        self.fc_out = nn.Linear(hidden_dim, output_dim) # 3: To make predictions for next token
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size] # seq length is 1  
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        # initial hidden, cell state is passed from encoder
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        # after sqeeze output: [batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0)) # remove all dimensions of size 0
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

# %% [markdown]
# The seq2seq model is built now. It will use both the encoder and the decoder. Encoder will generate the context vector while decover will be used to generate the target. 
# 
# Output from each hidden layer in the Encoder will be fed to the Decoder. Hence for simplicity, the number of layers need to be same in Encoder and Decoder. However, the layers can be different by using average of layers etc.. For example if Encoder has 2 layers and Decoder 1, then the average of 2 layers can be taken to reduce the dimensionality to 1, and then pass the values on. Similarly, both Encoder and decoder should have same number of dimensions.
# 
# The routine also used teacher forcing. Teacher forcing inserts the actual token (ground truth) and not the predicted token. Teacher forcing is controlled by a probability thereshold, which determines if the next token should be given by prediction or the actual value should be passed.

# %%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        print("src size:", src.size())
        # src = [src length, batch size]
        # trg = [trg length, batch size] # target
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # hidden, cell are from the output of encoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

# %% [markdown]
# Initialize the model before training.

# %%
# Translation is from German to English
input_dim = len(de_vocab)  # de_vocab stores token_ids
output_dim = len(en_vocab) # en_vocab["my"] = 1916, stores token ids
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("*** Using device:",device)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

# define the model
model = Seq2Seq(encoder, decoder, device).to(device)

# %% [markdown]
# Initialize the weights, using a uniform distribution. Also define a count_params function that will count the number of params in our model.

# %%
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


# Using required_grad ensures only trainable params are used in the count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% [markdown]
# Define the optimizer and loss function

# %%
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

# %% [markdown]
# Now add the main training function, this will take in model, optimizer, loss function and other details and execute the training loop. The training loop is called for each epoch. Returns the avg. epoch_loss.

# %%
def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        print("de batch:",batch["de_ids"])
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        # Send the token ids to the model
        # This will run the seq2seq model
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1] # Get the value from last index
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# %% [markdown]
# Create a separate function to evaluate the results. Similar to training, but with model.eval() enabled and no use of optim function. Return the avg. epoch loss for each batch, will be called for each epoch.

# %%
def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# %% [markdown]
# Start the model training. This will be the orchestrator that will call all other prev. defined functions.

# %%
n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

# model is already defined before in line 55
# batch_size is define in line 51

print(f"Batch Size {batch_size}")
# tqdm will auto display a progress bar
for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt") # Save the model with best loss
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")


