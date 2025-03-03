import numpy as np
import matplotlib.pyplot as plt

# Creiamo un dataset di esempio: prezzi di case basati su metratura e numero di stanze
np.random.seed(42)  # Per riproducibilità

# Numero di esempi
n_samples = 20

# Feature: metratura (m²) e numero di stanze
X = np.zeros((n_samples, 2))  # Matrice di input: 20 esempi, 2 feature ciascuno
X[:, 0] = np.random.randint(50, 200, n_samples)  # Metratura tra 50 e 200 m²
X[:, 1] = np.random.randint(1, 6, n_samples)  # Numero di stanze tra 1 e 5

# Parametri reali che vogliamo scoprire (nell'esempio, il modello "vero" sarebbe prezzo = 1.5*metratura + 30*stanze + 20)
true_w = np.array([1.5, 30])  # Coefficienti: 1.5 k€/m² e 30 k€/stanza
true_b = 20  # Intercetta: 20 k€ base

# Target: prezzo in migliaia di euro
y = np.dot(X, true_w) + true_b + np.random.normal(0, 10, n_samples)  # Aggiungiamo un po' di rumore

# Iniziamo con parametri casuali
w = np.random.randn(2)  # [w_metratura, w_stanze]
b = 0

# Iperparametri
learning_rate = 0.0001
epochs = 100

# Storia dei parametri e perdite per visualizzazione
history = {
    'w1': [w[0]],  # w per metratura
    'w2': [w[1]],  # w per stanze
    'b': [b],
    'loss': []
}

print("Dataset:")
print("X (prime 5 righe):")
print("   Metratura  Stanze")
for i in range(5):
    print(f"   {X[i, 0]:.1f}       {X[i, 1]:.0f}")
print("\ny (primi 5 prezzi):", y[:5])
print("\nParametri veri da scoprire:")
print(f"w_metratura = {true_w[0]}, w_stanze = {true_w[1]}, b = {true_b}")
print("\nParametri iniziali:")
print(f"w_metratura = {w[0]:.4f}, w_stanze = {w[1]:.4f}, b = {b:.4f}")

print("\nTraining del modello:")
print("Epoch   Loss      w_metratura  w_stanze   b")
print("-" * 48)

# Training loop
for epoch in range(epochs):
    # Forward pass (calcolo predizione) usando notazione matriciale
    y_pred = np.dot(X, w) + b

    # Calcolo errore (loss)
    loss = np.mean((y_pred - y) ** 2)  # Mean Squared Error

    # Calcolo gradiente
    dw = 2 * np.dot(X.T, (y_pred - y)) / n_samples  # X.T è la trasposta di X
    db = 2 * np.mean(y_pred - y)

    # Aggiornamento parametri
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Salva i parametri per visualizzazione
    history['w1'].append(w[0])
    history['w2'].append(w[1])
    history['b'].append(b)
    history['loss'].append(loss)

    # Stampa alcune epoche per vedere l'evoluzione
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"{epoch:5d}   {loss:8.2f}   {w[0]:8.4f}   {w[1]:8.4f}   {b:8.4f}")

print("\nParametri finali:")
print(f"w_metratura = {w[0]:.4f}, w_stanze = {w[1]:.4f}, b = {b:.4f}")
print(f"Parametri veri: w_metratura = {true_w[0]}, w_stanze = {true_w[1]}, b = {true_b}")

# Visualizzazione dell'evoluzione dei parametri
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history['w1'])
plt.axhline(y=true_w[0], color='r', linestyle='--')
plt.title('Evoluzione w_metratura')
plt.xlabel('Epoch')
plt.ylabel('w_metratura')

plt.subplot(2, 2, 2)
plt.plot(history['w2'])
plt.axhline(y=true_w[1], color='r', linestyle='--')
plt.title('Evoluzione w_stanze')
plt.xlabel('Epoch')
plt.ylabel('w_stanze')

plt.subplot(2, 2, 3)
plt.plot(history['b'])
plt.axhline(y=true_b, color='r', linestyle='--')
plt.title('Evoluzione b')
plt.xlabel('Epoch')
plt.ylabel('b')

plt.subplot(2, 2, 4)
plt.plot(history['loss'])
plt.title('Evoluzione Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.yscale('log')

plt.tight_layout()
plt.show()

print("\nDimostrazione del calcolo matriciale per la prima epoca:")
print("1. X (prime 3 righe della matrice di input):")
print(X[:3])

print("\n2. w iniziale (vettore dei pesi):")
print(w)

print("\n3. Calcolo della predizione y_pred = X·w + b:")
y_pred_demo = np.dot(X[:3], w) + b
print(f"X·w per le prime 3 righe:")
for i in range(3):
    print(f"({X[i, 0]} × {w[0]:.4f}) + ({X[i, 1]} × {w[1]:.4f}) + {b:.4f} = {y_pred_demo[i]:.4f}")

print("\n4. Calcolo dell'errore (y_pred - y):")
error_demo = y_pred_demo - y[:3]
for i in range(3):
    print(f"{y_pred_demo[i]:.4f} - {y[:3][i]:.4f} = {error_demo[i]:.4f}")

print("\n5. Calcolo del gradiente:")
print("dw = 2 * X.T · (y_pred - y) / n_samples")
print("Per w_metratura:")
dw1_detail = 2 * sum([X[i, 0] * error_demo[i] for i in range(3)]) / 3
print(
    f"2 * ({X[0, 0]} × {error_demo[0]:.4f} + {X[1, 0]} × {error_demo[1]:.4f} + {X[2, 0]} × {error_demo[2]:.4f}) / 3 = {dw1_detail:.4f}")

print("Per w_stanze:")
dw2_detail = 2 * sum([X[i, 1] * error_demo[i] for i in range(3)]) / 3
print(
    f"2 * ({X[0, 1]} × {error_demo[0]:.4f} + {X[1, 1]} × {error_demo[1]:.4f} + {X[2, 1]} × {error_demo[2]:.4f}) / 3 = {dw2_detail:.4f}")

print("\n6. Aggiornamento dei parametri:")
w_new1 = w[0] - learning_rate * dw1_detail
w_new2 = w[1] - learning_rate * dw2_detail
print(f"w_metratura_new = {w[0]:.4f} - {learning_rate} × {dw1_detail:.4f} = {w_new1:.4f}")
print(f"w_stanze_new = {w[1]:.4f} - {learning_rate} × {dw2_detail:.4f} = {w_new2:.4f}")