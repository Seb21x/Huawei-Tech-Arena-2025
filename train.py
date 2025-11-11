import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from model import SentenceEncoder
import math

class SentenceDataset(Dataset):
    """
    Custom Dataset class.
    It loads sentence vector data from .dat files and corresponding labels.
    """

    def __init__(self, data_folder="data"):
        path_vectors1 = os.path.join(data_folder, "sentence_pack_1.dat")
        path_vectors2 = os.path.join(data_folder, "sentence_pack_2.dat")
        path_labels = os.path.join(data_folder, "label_pack.dat")

        self.vectors1 = self._load_vectors(path_vectors1)
        self.vectors2 = self._load_vectors(path_vectors2)

        with open(path_labels, "r") as f:
            self.labels = torch.tensor([float(line.strip()) for line in f], dtype=torch.float32)

        print(f"Dataset loaded! {len(self.labels)} samples")

    def _load_vectors(self, path):
        # Helper to read the custom .dat format
        all_vectors = []
        with open(path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split(' ')))
                num_of_tokens = int(parts[0])
                embedding_dim = int(parts[1])
                vector_data = torch.tensor(parts[2:], dtype=torch.float32)
                reshaped_vector = vector_data.view(num_of_tokens, embedding_dim)
                all_vectors.append(reshaped_vector)
        return all_vectors

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Fetches one sample (a pair of sentences and their label)
        return self.vectors1[idx], self.vectors2[idx], self.labels[idx]


def collate_fn_with_padding(batch):
    """
    Collate Function for the DataLoader.
    Concept: Sentences have different lengths. To process them in a batch,
    we must "pad" the shorter ones with zeros so all sentences in the
    batch have the same length.
    """
    vectors1, vectors2, labels = zip(*batch)

    # torch.nn.utils.rnn.pad_sequence does this padding automatically
    padded_vectors1 = torch.nn.utils.rnn.pad_sequence(vectors1, batch_first=True, padding_value=0)
    padded_vectors2 = torch.nn.utils.rnn.pad_sequence(vectors2, batch_first=True, padding_value=0)

    labels = torch.stack(labels)
    return padded_vectors1, padded_vectors2, labels


def nt_xent_loss(embeddings, temperature=0.05):
    """
    NT-Xent Loss (Contrastive Learning).

    Concept: This loss function pulls "positive" pairs (e.g., two views of
    the same sample, or in our case, emb1 and emb2 from the same pair)
    together, while pushing all other "negative" samples in the batch apart.
    The 'temperature' controls the "sharpness" of this push/pull.
    """
    device = embeddings.device
    N2 = embeddings.shape[0]  # N2 = 2 * BatchSize

    # Handle the last batch which might be smaller and odd
    if N2 % 2 != 0:
        embeddings = embeddings[:-1]
        N2 -= 1

    N = N2 // 2

    # Calculate similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask to remove self-similarity (diagonal)
    diag_mask = torch.eye(N2, device=device, dtype=torch.bool)
    sim_masked = sim.masked_fill(diag_mask, -1e9)

    # Find the positive pairs (emb_i vs emb_i+N and emb_i+N vs emb_i)
    pos_indices = torch.arange(N, device=device)
    pos1 = sim[pos_indices, pos_indices + N]
    pos2 = sim[pos_indices + N, pos_indices]
    positives = torch.cat([pos1, pos2])

    # LogSumExp trick for numerical stability
    log_denom = torch.logsumexp(sim_masked, dim=1)

    # The loss is -log( (exp(positive_sim)) / (sum(exp(all_other_sims))) )
    loss = log_denom - positives
    return loss.mean()


def hard_negative_margin_loss(anchor_embeddings, positive_embeddings, all_embeddings, margin=0.5):
    """
    Hard Negative Margin Loss.

    Concept: For each "anchor" embedding, it finds the "hardest" negative
    embedding in the batch (the one most similar to the anchor). It then
    tries to push this hard negative *further* away from the anchor than
    the "positive" embedding is, by at least a 'margin'.
    """
    # Find similarity of all anchors to all possible negatives
    sim_matrix = torch.matmul(anchor_embeddings, all_embeddings.T)

    # Mask out the "true" positive pair (diagonal)
    diag_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.bool)
    sim_matrix.masked_fill_(diag_mask, -1e9)

    # Find the hardest negative (max similarity)
    hardest_neg_sim = sim_matrix.max(dim=1)[0]

    # Calculate similarity of the actual positive pair
    pos_sim = (anchor_embeddings * positive_embeddings).sum(dim=1)

    # Hinge Loss
    hinge = F.relu(margin + hardest_neg_sim - pos_sim)
    return hinge.mean()


def cosine_similarity_loss(embeddings_1, embeddings_2, labels):
    """
    Cosine Similarity Loss (Regression).
    """
    # Calculate cosine similarity (-1 to 1)
    cos_sim = F.cosine_similarity(embeddings_1, embeddings_2)

    # Scale similarity to 0-1 range
    cos_sim_scaled = (cos_sim + 1.0) / 2.0

    # Scale labels to 0-1 range
    labels_scaled = labels / 5.0

    # Calculate Mean Squared Error
    mse = F.mse_loss(cos_sim_scaled, labels_scaled)
    return mse


class CosineAnnealingWarmupLR:
    """
    Custom Learning Rate Scheduler.

    Concept: Manages the learning rate over epochs.
    1. Warmup: Linearly increases LR from 0 to 'base_lr' for 'warmup_epochs'.
    2. Cosine Annealing: Smoothly decreases LR from 'base_lr' to 'min_lr'
       following a cosine curve for the remaining epochs.
    This helps stabilize training at the beginning and find a better
    minimum at the end.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            # Ensure LR doesn't drop below min_lr
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
        return self.optimizer.param_groups[0]['lr']


def main():
    DEVICE = torch.device("cpu")

    # Set thread count
    torch.set_num_threads(1)

    # HYPERPARAMETERS
    BATCH_SIZE = 64
    WARMUP_EPOCHS = 3
    TOTAL_EPOCHS = 100
    CONTRASTIVE_PHASE_EPOCHS = 20
    INITIAL_LEARNING_RATE = 0.0004
    WEIGHT_DECAY = 0.00005
    MIN_LEARNING_RATE = 1e-6
    GRAD_ACCUMULATION_STEPS = 1
    HARD_NEGATIVE_MINING_FREQUENCY = 2

    # loading data

    full_dataset = SentenceDataset(data_folder="data")

    train_set_size = int(0.8 * len(full_dataset))
    validation_set_size = len(full_dataset) - train_set_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_set_size, validation_set_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
        num_workers=0,
        pin_memory=False
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_with_padding,
        num_workers=0,
        pin_memory=False
    )

    model = SentenceEncoder(output_size=256, hidden_size=128)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    learning_rate_scheduler = CosineAnnealingWarmupLR(
        optimizer, WARMUP_EPOCHS, TOTAL_EPOCHS, MIN_LEARNING_RATE
    )

    best_validation_loss = np.inf

    print(f"🚀 STARTING TRAINING")
    print(f" Contrastive Phase: {CONTRASTIVE_PHASE_EPOCHS} epochs")
    print(f" Fine-tune Phase: {TOTAL_EPOCHS - CONTRASTIVE_PHASE_EPOCHS} epochs")

    # training loop

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        is_contrastive_phase = (epoch < CONTRASTIVE_PHASE_EPOCHS)
        current_learning_rate = learning_rate_scheduler.step(epoch)

        for batch_index, (sentence_1_batch, sentence_2_batch, labels_batch) in enumerate(train_dataloader):
            sentence_1_batch = sentence_1_batch.to(DEVICE)
            sentence_2_batch = sentence_2_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            # Forward Pass
            embeddings_1 = model(sentence_1_batch)
            embeddings_2 = model(sentence_2_batch)

            # Loss Calculation
            if is_contrastive_phase:
                combined_embeddings = torch.cat([embeddings_1, embeddings_2], dim=0)
                batch_loss = nt_xent_loss(combined_embeddings, temperature=0.05)
            else:
                mse_loss_value = cosine_similarity_loss(embeddings_1, embeddings_2, labels_batch)
                if batch_index % HARD_NEGATIVE_MINING_FREQUENCY == 0:
                    hard_negative_loss_value = hard_negative_margin_loss(
                        embeddings_1, embeddings_2, embeddings_2, margin=0.4
                    )
                    batch_loss = mse_loss_value + hard_negative_loss_value
                else:
                    batch_loss = mse_loss_value

            batch_loss = batch_loss / GRAD_ACCUMULATION_STEPS

            # Backward Pass
            batch_loss.backward()

            # Optimizer Step
            if (batch_index + 1) % GRAD_ACCUMULATION_STEPS == 0:
                # Clip gradients to prevent them from exploding
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()
                # Clear gradients
                optimizer.zero_grad()

            epoch_train_loss += batch_loss.item() * GRAD_ACCUMULATION_STEPS
            num_train_batches += 1

        average_train_loss = epoch_train_loss / max(1, num_train_batches)
        phase_name = "Contrastive" if is_contrastive_phase else "Fine-tune"
        print(
            f"Epoch {epoch + 1:3d}/{TOTAL_EPOCHS} ({phase_name:11s}) | LR: {current_learning_rate:.6f} | Train Loss: {average_train_loss:.4f}",
            end=" | ")

        # VALIDATION LOOP
        model.eval()
        epoch_validation_loss = 0.0
        num_validation_batches = 0

        with torch.no_grad():
            for batch_index, (sentence_1_batch, sentence_2_batch, labels_batch) in enumerate(validation_dataloader):
                sentence_1_batch = sentence_1_batch.to(DEVICE)
                sentence_2_batch = sentence_2_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)

                # Forward Pass
                embeddings_1 = model(sentence_1_batch)
                embeddings_2 = model(sentence_2_batch)

                if is_contrastive_phase:
                    combined_embeddings = torch.cat([embeddings_1, embeddings_2], dim=0)
                    batch_validation_loss = nt_xent_loss(combined_embeddings, temperature=0.05).item()
                else:
                    mse_val_loss = cosine_similarity_loss(embeddings_1, embeddings_2, labels_batch).item()
                    if num_validation_batches == 0:
                        hn_val_loss = hard_negative_margin_loss(
                            embeddings_1, embeddings_2, embeddings_2, margin=0.4
                        ).item()
                    else:
                        hn_val_loss = 0.0
                    batch_validation_loss = mse_val_loss + hn_val_loss

                epoch_validation_loss += batch_validation_loss
                num_validation_batches += 1

        average_validation_loss = epoch_validation_loss / max(1, num_validation_batches)
        print(f"Validation Loss: {average_validation_loss:.4f}", end="")

        # checkpoint
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), "model.bin")
            print(f" ✅ NEW BEST MODEL SAVED!")
        else:
            print(f" (Best: {best_validation_loss:.4f})")

    print("✅ TRAINING COMPLETE!")


if __name__ == '__main__':
    main()