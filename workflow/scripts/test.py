import numpy as np

# Beispiel: Dummy-Daten für die Summen
plus_plus = np.array([[1, 2, 3], [4, 5, 6]])  # Array mit Zykluswerten
pos = np.array([[1, 0, 2], [3, 2, 1]])
neg = np.array([[0, 2, 1], [1, 0, 3]])

# Zykluslängen k (1-basiert, da 1/k gewichtet wird)
k = np.arange(1, plus_plus.shape[1] + 1)

# Gewichtung 1/k
weights = 1 / k

print(weights)

# Gewichtete Summen berechnen
weighted_total_sum_cycles = np.nansum(plus_plus * weights, axis=1)
weighted_pos_sum_cycles = np.nansum(pos * weights, axis=1)
weighted_neg_sum_cycles = np.nansum(neg * weights, axis=1)

# Ergebnisse ausgeben
print("Gewichtete Total-Summen:", weighted_total_sum_cycles)
print("Gewichtete Positive-Summen:", weighted_pos_sum_cycles)
print("Gewichtete Negative-Summen:", weighted_neg_sum_cycles)
