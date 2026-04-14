import numpy as np 
import re
import os
import logging
import random
import torch 
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
import math
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""Set up logging system to record timestamped outputs in seperate text file"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_results.txt"), 
        logging.StreamHandler()                        
    ]
)

"""Set randomness for reproducibility"""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

"""Organise data into lists of reviews with corresponding sentiment labels"""
reviews = []
sentiment_ratings = []

with open("Compiled_Reviews.txt", encoding="utf-8") as f:
    next(f) 
   
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) >= 2:
            reviews.append(fields[0])
            sentiment_ratings.append(fields[1])

"""Load word2vec model for embeddings"""
from gensim.models import KeyedVectors
model_path = 'GoogleNews-vectors-negative300.bin'
w = KeyedVectors.load_word2vec_format(model_path, binary=True)

"""
Clean text in reviews:
- retain apostrophes and hypens which are present in word2vec
- lower case for embedding look-up consistenecy  
"""
tokenized_revs = [re.findall(r"[A-Za-z0-9'\-]+", rev.lower()) for rev in reviews]

"""
Create an averaged embedding vector for each review using word2vec
These are used for the logistic regression model
"""
embeddings = []

for tokens in tokenized_revs:
    vecs = [w[t] for t in tokens if t in w] 

    if vecs:
        rev_vec = np.mean(vecs, axis=0)
    else:
        rev_vec = np.zeros(w.vector_size) 

    embeddings.append(rev_vec)

"""Stack vectors vertically to create input embeddings matrix"""
embeddings = np.vstack(embeddings)

"""Generate training, development and test indices for both models"""
indices = list(range(len(tokenized_revs)))
random.shuffle(indices)


train_size = int(0.8 * len(indices))
remaining_size = len(indices) - train_size
dev_size = remaining_size // 2

train_idx = indices[:train_size]
dev_idx = indices[train_size : train_size + dev_size]
test_idx = indices[train_size + dev_size:]

"""
Create DataLoader objects for logistic regression model using embeddings matrix 
and train, dev, test indices 
"""
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)


labels_list = [1.0 if l == "positive" else 0.0 for l in sentiment_ratings]
labels_tensor = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
full_dataset_lr = TensorDataset(embeddings_tensor, labels_tensor)

train_dataset_lr = Subset(full_dataset_lr, train_idx)
dev_dataset_lr   = Subset(full_dataset_lr, dev_idx)
test_dataset_lr  = Subset(full_dataset_lr, test_idx)


train_loader_lr = DataLoader(train_dataset_lr, batch_size=32, shuffle=True)
dev_loader_lr   = DataLoader(dev_dataset_lr, batch_size=32, shuffle=False)
test_loader_lr  = DataLoader(test_dataset_lr, batch_size=32, shuffle=False)

"""Initialise logistic regression model"""
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        output = self.linear(x)
        return output

"""Create an instance of logistic regression model"""
input_dim = 300
model_1 = LogisticRegressionModel(input_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

"""Training loop for logistic regression model"""
"""Set up early stopping logic"""
patience = 5
patience_counter = 0
best_dev_acc = 0.0 
model_1_save_path = "best_logistic_model.pth"

num_epochs = 250
logging.info("Logistic regression model training starting...")
logging.info("-" * 30)

for epoch in range(num_epochs):
    model_1.train()
    total_loss = 0

    for batch_embeddings, batch_labels in train_loader_lr:
        batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
        
        outputs = model_1(batch_embeddings)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    """ Evaluation using development set """
    model_1.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for dev_embeddings, dev_labels in dev_loader_lr:
            dev_embeddings, dev_labels = dev_embeddings.to(device), dev_labels.to(device)
            
            outputs = model_1(dev_embeddings)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            correct += (preds == dev_labels).sum().item()
            total += dev_labels.size(0)
    

    dev_acc = correct / total    
    avg_loss = total_loss / len(train_loader_lr)
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Training Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    """ Implement early stopping logic based on Accuracy """
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model_1.state_dict(), model_1_save_path)
        patience_counter = 0  
    else:
        patience_counter += 1

    if patience_counter >= patience:
        logging.info(f"Early stopping triggered at epoch {epoch+1}.")
        break

logging.info(f"TRAINING COMPLETE")
logging.info("-" * 30)
logging.info(f"Best Dev Acc: {best_dev_acc:.4f}")
logging.info(f"Best Model saved to {model_1_save_path}")
logging.info("-" * 30)

"""Preprocess reveiws for transformer model"""
class ReviewDataset(Dataset):
    def __init__(self, tokenized_reviews, labels, w_model, max_len=512):
        self.data = tokenized_reviews
        self.labels = labels
        self.w = w_model
        self.max_len = max_len
        self.vec_size = w_model.vector_size

        """
        Args:
            tokenized_reviews: list of token lists for reviews
            labels: target sentiment labels aligned with reviews
            w_model: pretrained word2vec embedding model
            max_len: maximum sequence length expected by the transformer
        """

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        label = self.labels[idx]

        """
        Retrieve embedding vector for each word in the review 
        Truncate reviews to max_length expected by transformer
        """
        vecs = [self.w[t] for t in tokens if t in self.w]
        vecs = vecs[:self.max_len]

        """Compute padding size if needed and add to vec"""
        if vecs:
            vecs = np.array(vecs)
            num_real_words = len(vecs) 
            pad_width = self.max_len - num_real_words

            if pad_width > 0:
                padding = np.zeros((pad_width, self.vec_size))
                vecs = np.vstack([vecs, padding])
            mask = [1] * num_real_words + [0] * pad_width
        
        else:
            vecs = np.zeros((self.max_len, self.vec_size))
            mask = [0]*self.max_len
            
        """Return as tensors"""
        label = torch.tensor(label, dtype=torch.float32)

        return torch.tensor(vecs, dtype=torch.float32), torch.tensor(mask, dtype=torch.long), label

"""
Generate DataLoader objects for transformer model using ReviewDataset preprocessing class
 and train, dev, test indices
 """
full_dataset_trans = ReviewDataset(tokenized_revs, labels_list, w)


train_dataset_trans = Subset(full_dataset_trans, train_idx)
dev_dataset_trans   = Subset(full_dataset_trans, dev_idx)
test_dataset_trans  = Subset(full_dataset_trans, test_idx)

trans_train_loader = DataLoader(train_dataset_trans, batch_size=32, shuffle=True)
trans_dev_loader   = DataLoader(dev_dataset_trans, batch_size=32, shuffle=False)
trans_test_loader  = DataLoader(test_dataset_trans, batch_size=32, shuffle=False)

class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 514, padding_idx: int = 1):
        """
        Args:
            d_model: hidden size
            max_len: max sequence length supported (often 512 + 2 specials)
            padding_idx: index in input_ids used for padding (RoBERTa typically 1)
        """
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        """
        +1 because index 0 is resvered for PAD so we need an additional index
        to reach maximum length
        """
        self.position_embeddings = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        """
        We'll map pad positions to index 0 in the position table.
        (i.e., row 0 stays zeroed and is never updated)
        """
        with torch.no_grad():
            self.position_embeddings.weight[0].zero_()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        position_ids = (torch.cumsum(mask, dim=1) * mask).to(device).to(torch.long)

        pos_emb = self.position_embeddings(position_ids)
        return x + pos_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        """
        Args:
                input_dim: embedding size '300'
                num_heads: number of attention heads
        """
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads


        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

        """Combines all heads back into a single representation"""
        self.out_projection = nn.Linear(input_dim, input_dim)

    def forward(self, X, mask = None, causal_masking=False):
        batch_size, seq_len, _ = X.shape

        """Generate queries, keys and values using learnable linear projections"""
        Q = self.query_projection(X)
        K = self.key_projection(X)
        V = self.value_projection(X)

        """
        Split Q, K, V into a different view for each attention head
        Dim: (batch_size, num_heads, seq_len, head_dim)
        """
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     
        
        """
        Calculate attention scores:
        Dim: (batch, num_heads, seq_len, seq_len)
        """
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        """Apply padding mask if needed"""
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(mask == 0, float("-inf"))

        """Apply causal masking if set to True"""
        if causal_masking:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        """Calculate attention weights using softmax over scores and combine with values"""
        attention_weights = torch.softmax(scores, dim=-1)
        context = attention_weights @ V
        
        """
        Dims transformation:

        Initial:
        (batch_size, num_heads, seq_len, head_dim)

        1. Move sequence dimension before heads
        → (batch_size, seq_len, num_heads, head_dim)

        2. Concatenate attention heads (num_heads × head_dim)
        → (batch_size, seq_len, input_dim)

        3. Apply learned output projection to mix information across heads
        """
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, seq_len, self.input_dim)
        return self.out_projection(context)

class LayerNorm(nn.Module):
    """
    BERT-style LayerNorm: normalise over last dimension with learnable
    scale (gamma) and bias (beta).
    """
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            torch.var(x, dim=-1, correction=0, keepdim=True)  + self.eps
        ) * self.weight + self.bias

class MLP(nn.Module):
    """
    MLP, applied elementwise over the last dimension
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fully_connected1 = nn.Linear(input_dim, hidden_dim)
        self.fully_connected2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.everything = nn.Sequential(
            self.fully_connected1,
            self.relu,
            self.fully_connected2
        )

    def forward(self, X):
        return self.everything(X)

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, mlp_hidden_dim, dropout_p=0.1):
        super().__init__()

        """Attention block with dropout"""
        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_p)

        """MLP block with dropout"""
        self.norm2 = LayerNorm(input_dim)
        self.mlp = MLP(input_dim, mlp_hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)

    """Causal masking set to False for encoder design"""
    def forward(self, X, mask=None, causal_masking=False):
        attn_out = self.attention(
            self.norm1(X),
            mask=mask,
            causal_masking=causal_masking
        )
        X = X + self.dropout1(attn_out)

        mlp_out = self.mlp(self.norm2(X))
        X = X + self.dropout2(mlp_out)

        return X

class ClassificationHead(nn.Module):
    """
    Final classifier head for encoder outputs
    """
    def __init__(self, input_dim, num_classes, dropout_p=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, mask=None):
        """ 
        Mean pool along the sequence length dimension
        Apply padding mask, using .clamp() to avoid division by zero
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class EncoderModel(nn.Module):
    """
    Combine positional embeddings, transformer layer and classification head into one model
    """
    def __init__(
        self,
        input_dim,
        num_heads,
        mlp_hidden_dim,
        num_layers,
        num_classes=1,
        max_length=512,
        dropout_p=0.1,
):
        super().__init__()

        self.positional_embedding = LearnedPositionalEmbedding(
            d_model=input_dim,
            max_len=max_length
        )

        self.layers = nn.ModuleList([
            TransformerLayer(
                input_dim=input_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout_p=dropout_p
            )
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(input_dim)

        self.classifier = ClassificationHead(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout_p=dropout_p
            )

    def forward(self, x, mask):
        x = self.positional_embedding(x, mask)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.classifier(x, mask)

        return(logits)

"""Create an instance of encoder model"""
input_dim = 300
num_heads = 6
mlp_hidden_dim = 512
num_layers = 2
num_classes = 1  
dropout_p = 0.1
max_length = 512

model_2 = EncoderModel(
    input_dim=input_dim,
    num_heads=num_heads,
    mlp_hidden_dim=mlp_hidden_dim,
    num_layers=num_layers,
    num_classes=num_classes,
    max_length=max_length,
    dropout_p=dropout_p
)

model_2.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_2.parameters(), lr=1e-4)

"""Training loop for Transformer model"""
"""Set up early stopping logic """
"""Set up early stopping logic"""
patience_2 = 5            
patience_counter_2 = 0     
best_dev_acc_2 = 0.0
model_2_save_path = "best_transformer_encoder.pth"   

num_epochs = 50
logging.info("Transformer encoder model training starting...")
logging.info("-" * 30)

for epoch in range(num_epochs):
    model_2.train()
    total_loss = 0

    for batch in trans_train_loader:
        x, mask, labels = batch
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()
   
        logits = model_2(x, mask).squeeze(-1)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(trans_train_loader)

    """ Evaluation using development set """
    model_2.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_dev, mask_dev, labels_dev in trans_dev_loader:
            x_dev, mask_dev, labels_dev = x_dev.to(device), mask_dev.to(device), labels_dev.to(device)
            
            logits_dev = model_2(x_dev, mask_dev).squeeze(-1)
            probs = torch.sigmoid(logits_dev)
            preds = (probs > 0.5).float()
            
            correct += (preds == labels_dev).sum().item()
            total += labels_dev.size(0)
    
    dev_acc = correct / total
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Training Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    """ Implement early stopping logic based on Accuracy (no min_improvement) """
    if dev_acc > best_dev_acc_2:
        best_dev_acc_2 = dev_acc
        torch.save(model_2.state_dict(), model_2_save_path)
        patience_counter_2 = 0  
    else:
        patience_counter_2 += 1

    if patience_counter_2 >= patience_2:
        logging.info(f"Early stopping triggered at epoch {epoch+1}.")
        break

logging.info(f"TRAINING COMPLETE")
logging.info("-" * 30)
logging.info(f"Best Dev Acc: {best_dev_acc_2:.4f}")
logging.info(f"Best Model saved to {model_2_save_path}")
logging.info("-" * 30)

def get_probabilities(model, loader):

    model.to(device)
    model.eval()

    pred_probs = []
    gold_labels = []

    with torch.no_grad():
        for batch in loader:

            """ Handle different data shapes for transformer and logistic regression models """
            if len(batch) == 3:
                bx, b_mask, by = batch
                bx = bx.to(device)
                b_mask = b_mask.to(device)
                by = by.to(device)
                logits = model(bx, b_mask)
            
            else:
                bx, by = batch
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)

            probs = torch.sigmoid(logits).squeeze(-1)

            pred_probs.append(probs.cpu().numpy())
            gold_labels.append(by.cpu().numpy())

    y_scores = np.concatenate(pred_probs)
    y_true = np.concatenate(gold_labels)

    return y_true, y_scores

def get_auc(model, loader):
    y_true, y_scores = get_probabilities(model, loader)
    return roc_auc_score(y_true, y_scores)

def bootstrap_auc(y_true, probs_1, probs_2, n_bootstrap=2000):
    auc_1 = roc_auc_score(y_true, probs_1)
    auc_2 = roc_auc_score(y_true, probs_2)
    difference = auc_2 - auc_1

    bootstrap_diffs = []
    indices = np.arange(len(y_true))

    for i in range(n_bootstrap):
        bootstrap_indices = resample(indices, replace=True, n_samples=len(y_true))
        
        """Compare AUC scores across models for all bootstrapped samples of test set"""
        boot_auc1 = roc_auc_score(y_true[bootstrap_indices], probs_1[bootstrap_indices])
        boot_auc2 = roc_auc_score(y_true[bootstrap_indices], probs_2[bootstrap_indices])
        bootstrap_diffs.append(boot_auc2-boot_auc1)

    bootstrap_diffs = np.array(bootstrap_diffs)

    """
    Calculate a 'null' difference expectation under null hypothesis and
    compare to observed difference
    """
    null = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = np.mean(null >= difference)

    return difference, p_value, bootstrap_diffs

"""
Generate predicted probabilities on test set according to both models
Pass these to boostrapping function to generate a p-value for difference between AUC scores
"""
model_1.load_state_dict(torch.load("best_logistic_model.pth", map_location=device))
model_2.load_state_dict(torch.load("best_transformer_encoder.pth", map_location=device))

y_true, probs_1 = get_probabilities(model_1, test_loader_lr)
y_true, probs_2 = get_probabilities(model_2, trans_test_loader)
difference, p_value, bootsrap_diff = bootstrap_auc(y_true, probs_1, probs_2)

logging.info(f"Logistic Regression Model AUC: {roc_auc_score(y_true, probs_1):.4f}")
logging.info(f"Encoder Transformer Model AUC: {roc_auc_score(y_true, probs_2):.4f}")
logging.info(f"P-value: {p_value:.4f}")