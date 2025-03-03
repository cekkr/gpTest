import numpy as np
import matplotlib.pyplot as plt

# Esempio di calcolo statico di una struttura a telaio semplice con 2 elementi
# ------------------------------------------------------------------------------

# 1. Definizione della geometria e delle proprietà dei materiali
# --------------------------------------------------------------
# Consideriamo un telaio a L rovesciata con un elemento verticale e uno orizzontale
# Coordinate dei nodi (in metri)
nodi = np.array([
    [0, 0],  # Nodo 1: incastro alla base
    [0, 3],  # Nodo 2: giunzione tra gli elementi
    [4, 3]  # Nodo 3: estremità libera
])

# Definizione degli elementi
elementi = np.array([
    [1, 2],  # Elemento 1: da nodo 1 a nodo 2 (verticale)
    [2, 3]  # Elemento 2: da nodo 2 a nodo 3 (orizzontale)
])

# Proprietà del materiale e della sezione
E = 210e9  # Modulo elastico (Pa) - acciaio
I = 2e-4  # Momento d'inerzia (m^4)
A = 0.01  # Area della sezione (m^2)


# 2. Costruzione della matrice di rigidezza per ogni elemento
# -----------------------------------------------------------

def calcola_matrice_rigidezza_locale(L, E, I, A):
    """
    Calcola la matrice di rigidezza locale per un elemento trave
    nel sistema di coordinate locale (6x6)
    """
    # Termini della matrice di rigidezza per una trave con 6 gradi di libertà
    # (spostamenti e rotazioni alle estremità)
    k11 = (E * A) / L
    k22 = (12 * E * I) / (L ** 3)
    k23 = (6 * E * I) / (L ** 2)
    k33 = (4 * E * I) / L
    k36 = (2 * E * I) / L

    # Matrice di rigidezza locale
    k_locale = np.array([
        [k11, 0, 0, -k11, 0, 0],
        [0, k22, k23, 0, -k22, k23],
        [0, k23, k33, 0, -k23, k36],
        [-k11, 0, 0, k11, 0, 0],
        [0, -k22, -k23, 0, k22, -k23],
        [0, k23, k36, 0, -k23, k33]
    ])

    return k_locale


def calcola_matrice_trasformazione(x1, y1, x2, y2):
    """
    Calcola la matrice di trasformazione dal sistema locale al globale
    """
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    c = (x2 - x1) / L  # coseno
    s = (y2 - y1) / L  # seno

    # Matrice di trasformazione
    T = np.array([
        [c, s, 0, 0, 0, 0],
        [-s, c, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0],
        [0, 0, 0, -s, c, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    return T, L


# 3. Assemblaggio della matrice di rigidezza globale
# --------------------------------------------------

# Numero totale di gradi di libertà: 3 nodi x 3 DOF (2 traslazioni + 1 rotazione)
ndof_totali = 3 * 3  # 9 DOF totali
K_globale = np.zeros((ndof_totali, ndof_totali))

for i, elemento in enumerate(elementi):
    nodo1, nodo2 = elemento

    # Coordinare dei nodi dell'elemento
    x1, y1 = nodi[nodo1 - 1]
    x2, y2 = nodi[nodo2 - 1]

    # Calcola matrice di trasformazione e lunghezza
    T, L = calcola_matrice_trasformazione(x1, y1, x2, y2)

    # Calcola matrice di rigidezza nel sistema locale
    k_locale = calcola_matrice_rigidezza_locale(L, E, I, A)

    # Trasforma nel sistema globale
    k_globale = T.T @ k_locale @ T

    # Mappa dei gradi di libertà dell'elemento
    dof_map = []
    for node in [nodo1, nodo2]:
        # Per ogni nodo: 3 DOF (ux, uy, rotazione)
        start_dof = (node - 1) * 3
        dof_map.extend([start_dof, start_dof + 1, start_dof + 2])

    # Assemblaggio nella matrice globale
    for i, dof_i in enumerate(dof_map):
        for j, dof_j in enumerate(dof_map):
            K_globale[dof_i, dof_j] += k_globale[i, j]

# 4. Applicazione delle condizioni al contorno
# -------------------------------------------

# Nodo 1 è incastrato (ux=0, uy=0, rot=0)
vincoli = [0, 1, 2]  # DOF vincolati

# Vettore delle forze esterne
F = np.zeros(ndof_totali)
# Applicazione di una forza verticale di 10 kN al nodo 3
F[7] = -10000  # DOF 7 = spostamento verticale del nodo 3

# 5. Soluzione del sistema
# -----------------------

# Separazione dei DOF liberi e vincolati
dof_liberi = [i for i in range(ndof_totali) if i not in vincoli]

# Estrazione della parte della matrice relativa ai DOF liberi
K_liberi = K_globale[np.ix_(dof_liberi, dof_liberi)]
F_liberi = F[dof_liberi]

# Risoluzione del sistema lineare: K * u = F
U_liberi = np.linalg.solve(K_liberi, F_liberi)

# Ricostruzione del vettore completo degli spostamenti
U = np.zeros(ndof_totali)
for i, dof in enumerate(dof_liberi):
    U[dof] = U_liberi[i]

# 6. Calcolo delle reazioni vincolari
# ----------------------------------
R = K_globale @ U

# 7. Stampa dei risultati
# ----------------------
print("Spostamenti nodali:")
for i in range(3):  # Per ogni nodo
    print(f"Nodo {i + 1}:")
    print(f"  ux = {U[i * 3]:.6e} m")
    print(f"  uy = {U[i * 3 + 1]:.6e} m")
    print(f"  rot = {U[i * 3 + 2]:.6e} rad")

print("\nReazioni vincolari al nodo 1:")
print(f"  Rx = {R[0]:.2f} N")
print(f"  Ry = {R[1]:.2f} N")
print(f"  M = {R[2]:.2f} N·m")


# 8. Visualizzazione della struttura deformata
# -------------------------------------------
def visualizza_struttura(nodi, elementi, U, fattore_scala=100):
    plt.figure(figsize=(10, 8))

    # Struttura indeformata
    for elemento in elementi:
        nodo1, nodo2 = elemento
        x = [nodi[nodo1 - 1][0], nodi[nodo2 - 1][0]]
        y = [nodi[nodo1 - 1][1], nodi[nodo2 - 1][1]]
        plt.plot(x, y, 'k--', linewidth=1)

    # Struttura deformata
    nodi_deformati = nodi.copy()
    for i in range(len(nodi)):
        nodi_deformati[i][0] += U[i * 3] * fattore_scala
        nodi_deformati[i][1] += U[i * 3 + 1] * fattore_scala

    for elemento in elementi:
        nodo1, nodo2 = elemento
        x = [nodi_deformati[nodo1 - 1][0], nodi_deformati[nodo2 - 1][0]]
        y = [nodi_deformati[nodo1 - 1][1], nodi_deformati[nodo2 - 1][1]]
        plt.plot(x, y, 'r-', linewidth=2)

    # Nodi
    plt.plot(nodi[:, 0], nodi[:, 1], 'ko', markersize=8)

    # Forze
    if abs(F[7]) > 0:  # Se c'è una forza verticale al nodo 3
        plt.arrow(nodi[2][0], nodi[2][1], 0, -0.5,
                  head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        plt.text(nodi[2][0] + 0.1, nodi[2][1] - 0.3, f"{abs(F[7] / 1000):.1f} kN", color='blue')

    plt.grid(True)
    plt.axis('equal')
    plt.title("Struttura deformata (fattore di scala: " + str(fattore_scala) + ")")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    return plt


fig = visualizza_struttura(nodi, elementi, U)
plt.tight_layout()
plt.show()