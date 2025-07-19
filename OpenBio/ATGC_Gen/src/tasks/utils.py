import re, math
import torch
import numpy as np
from scipy.linalg import sqrtm
import scipy
from tqdm import tqdm
from itertools import product

valid_vocab = {
    'A': 7,
    'C': 8,
    'G': 9,
    'T': 10,
    'N': 11,
}
DNA_vocab = {
    'A': 7,
    'C': 8,
    'G': 9,
    'T': 10,
}
index_mapping = {
    valid_vocab['A']: 0,
    valid_vocab['C']: 1,
    valid_vocab['G']: 2,
    valid_vocab['T']: 3,
    valid_vocab['N']: 4,
}


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def convert_batch_one_hot(ids, fill=0.25):
    batch_size, seq_length = ids.shape
    num_classes = 4
    one_hot = torch.full((batch_size, seq_length, num_classes), fill).to(ids.device)

    for char, char_index in DNA_vocab.items():
        mask = (ids == char_index)
        one_hot_vector = torch.zeros(num_classes).to(ids.device)
        one_hot_vector[index_mapping[char_index]] = 1.0
        one_hot[mask] = one_hot_vector
        # one_hot[mask, self.index_mapping[char_index]] = 1.0

    return one_hot


def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist


# def get_wasserstein_dist(embeds1, embeds2):
#     if torch.isnan(embeds2).any() or torch.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
#         return torch.tensor(float('nan'))
#
#     mu1 = embeds1.mean(dim=0)
#     sigma1 = torch.cov(embeds1.T.float())
#     mu2 = embeds2.mean(dim=0)
#     sigma2 = torch.cov(embeds2.T.float())
#
#     ssdiff = torch.sum((mu1 - mu2) ** 2.0)
#     covmean = torch.sqrt(sigma1.mm(sigma2))
#
#     if torch.is_complex(covmean):
#         covmean = covmean.real
#
#     dist = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return dist


def calculate_diversity(sequences, n_min=10, n_max=12):
    """ Calculate the diversity of a list of DNA sequences """
    diversity_product = torch.tensor(1.0, dtype=torch.float32)
    diversity_sum = torch.tensor(0.0, dtype=torch.float32)
    sequences = [t.tolist() for t in sequences]

    for n in range(n_min, n_max + 1):
        n_grams = [tuple(seq[i:i + n]) for seq in sequences for i in range(len(seq) - n + 1)]
        total_n_grams = len(n_grams)
        unique_n_grams = len(set(n_grams))
        diversity = torch.tensor(unique_n_grams / total_n_grams, dtype=torch.float32)
        # diversity = total_n_grams / unique_n_grams if unique_n_grams > 0 else 0
        diversity_product *= diversity

        diversity_each_sum = torch.tensor(math.log(unique_n_grams / total_n_grams), dtype=torch.float32)
        diversity_sum += diversity_each_sum
    return diversity_product, diversity_sum


def calculate_weighted_category_diversity(categories):
    weighted_diversity_scores_prod = []
    weighted_diversity_scores_sum = []
    total_sequences = sum(len(sequences) for sequences in categories.values())

    for category, sequences in categories.items():
        diversity_prod, diversity_sum = calculate_diversity(sequences)
        weight = len(sequences) / total_sequences
        weighted_diversity_scores_prod.append(diversity_prod * weight)
        weighted_diversity_scores_sum.append(diversity_sum * weight)

    weighted_average_diversity_prod = torch.tensor(weighted_diversity_scores_prod).sum()
    # weighted_average_diversity_sum = sum(weighted_diversity_scores_sum)

    return weighted_average_diversity_prod


def percent_identity(batch_pred, batch_original):
    batch_pred = batch_pred[:, 1:-1]
    batch_original = batch_original[:, 1:-1]
    batch_pred = convert_batch_one_hot(batch_pred, fill=0.0)
    batch_original = convert_batch_one_hot(batch_original, fill=0.0)
    bs, seq_length, alphabet_size = batch_pred.shape

    batch_original = batch_original.reshape(-1, seq_length * alphabet_size)
    batch_pred = batch_pred.reshape(-1, seq_length * alphabet_size)

    batch_result = torch.mm(batch_original, batch_pred.T)
    diagonal_elements = torch.diag(batch_result)

    return diagonal_elements


def percent_identity_group(all_pred, all_original, all_class):
    pred_dict, original_dict = {}, {}
    for i in range(len(all_class)):
        cur_class = all_class[i].item()
        pred_dict.setdefault(cur_class, []).append(all_pred[i])
        original_dict.setdefault(cur_class, []).append(all_original[i])

    # iterate over different classes
    avg_pi_go, avg_pi_gg = [], []
    max_pi_go, max_pi_gg = 0, 0
    for class_type in pred_dict.keys():
        # generated vs original
        cur_pred = torch.stack(pred_dict[class_type], dim=0) if len(pred_dict[class_type]) > 1 else pred_dict[class_type][0].unsqueeze(0)
        cur_original = torch.stack(original_dict[class_type], dim=0) if len(original_dict[class_type]) > 1 else original_dict[class_type][0].unsqueeze(0)

        cur_pred = np.array(convert_batch_one_hot(cur_pred, fill=0.0))
        cur_original = np.array(convert_batch_one_hot(cur_original, fill=0.0))

        cur_pi_train = percent_identity_one_all(cur_pred, cur_original)
        cur_pi_train = np.max(cur_pi_train, axis=1)

        max_pi_go = max(max_pi_go, np.max(cur_pi_train))
        avg_pi_go.append(np.average(cur_pi_train))

        # generated vs generated
        if len(cur_pred) != 1:
            cur_pi_diversity = percent_identity_one_all(cur_pred, cur_pred)
            cur_diversity = []
            for i in range(len(cur_pi_diversity)):
                sort = np.sort(cur_pi_diversity[i])[::-1]
                cur_diversity.append(sort[1])
            cur_diversity = np.array(cur_diversity)

            max_pi_gg = max(max_pi_gg, np.max(cur_diversity))
            avg_pi_gg.append(np.average(cur_diversity))

    return (torch.tensor(max_pi_go, dtype=torch.float32),
            torch.tensor(np.average(avg_pi_go), dtype=torch.float32),
            torch.tensor(max_pi_gg, dtype=torch.float32),
            torch.tensor(np.average(avg_pi_gg), dtype=torch.float32))


def percent_identity_one_all(pred_array, original_array, bs=500):
    pred_size, seq_length, alphabet_size = pred_array.shape
    original_size = original_array.shape[0]

    pred_array = np.reshape(pred_array, [-1, seq_length * alphabet_size])
    original_array = np.reshape(original_array, [-1, seq_length * alphabet_size])

    seq_identity = np.zeros((pred_size, original_size)).astype(np.int8)

    for start_idx in range(0, pred_size, bs):
        end_idx = min(start_idx + bs, pred_size)
        batch_result = np.dot(pred_array[start_idx:end_idx], original_array.T)
        seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)

    return seq_identity


def kmer_statistics(data1, data2, kmer_length=7):
    data1 = convert_batch_one_hot(data1[:, 1:-1]).numpy()
    data2 = convert_batch_one_hot(data2[:, 1:-1]).numpy()
    # generate kmer distributions
    dist1 = compute_kmer_spectra(data1, kmer_length)
    dist2 = compute_kmer_spectra(data2, kmer_length)

    # computer KLD
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)

    # computer jensen-shannon
    jsd = np.round(np.sum(scipy.spatial.distance.jensenshannon(dist1, dist2)), 6)

    return torch.tensor(kld), torch.tensor(jsd)


def compute_kmer_spectra(
        X,
        kmer_length=3,
        dna_dict=None
):
    # convert one hot to A,C,G,T
    if dna_dict is None:
        dna_dict = {
            0: "A",
            1: "C",
            2: "G",
            3: "T"
        }
    seq_list = []

    for index in range(len(X)):
        # for loop is what actually converts a list of one-hot encoded sequences into ACGT
        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"],
                                                     repeat=kmer_length)]  # list of all kmer permutations, length specified by repeat=

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)

    # what to compute entropy against
    global_counts_normalized = global_counts / sum(global_counts)  # this is the distribution of kmers in the testset
    # print(global_counts_normalized)
    return global_counts_normalized


class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'C', 'G', 'T']
        self.multiplyBy = 4 ** np.arange(k - 1, -1,
                                         -1)  # the multiplying number for each digit position in the k-number system
        self.n = 4 ** k  # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = []  # a list containing the one-hot representation of kmers for each sequence in the list of sequences given
        for seq in seqs:  # first obtain the one-hot representation of the kmers in a sequence
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(),
                                                                          write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)  # append this one-hot list into another list

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):  #
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.
        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)  # array of zeroes the same length of all possible kmers

        for i in range(
                number_of_kmers):  # for each kmer feature, turn the corresponding index in the list of all kmer features to 1
            this_kmer = seq[i:(i + self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self,
                                    kmer):  # returns the corresponding index of a kmer in the larger list of all possible kmers?
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering
