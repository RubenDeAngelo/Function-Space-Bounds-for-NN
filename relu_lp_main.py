import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import logging
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import numpy as np

# --- Lokale Imports ---
from adaptive_lp_core import (
    ReLUNetwork,
    init_many_affine_pieces,
    AdaptiveLpNorm,
    monte_carlo_lp_norm
)


@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    """
    Unterdrückt Warnungen und Konsolen-Spam von externen Bibliotheken (z.B. auto_LiRPA).
    """
    prev_disable_level = logging.root.manager.disable
    prev_warning_filters = warnings.filters[:]

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            if disable_logging:
                logging.disable(logging.CRITICAL)  # Blockiert Logs
            if disable_warnings:
                warnings.simplefilter("ignore")  # Blockiert Warnungen
            yield
        finally:
            if disable_warnings:
                warnings.filters = prev_warning_filters
            if disable_logging:
                logging.disable(prev_disable_level)


def main():
    # ==========================================
    # 1. PARAMETER KONFIGURATION
    # ==========================================
    d = 2  # Eingabedimension
    p = 2.0  # Exponent der L_p Norm
    m = 30  # Anzahl der Adaptions-Iterationen
    theta = 0.2  # Dörfler Marking Parameter (0 < theta < 1)
    refiner = "max"  # Halbierungsstrategie ("max" oder "all")

    # ==========================================
    # 2. NETZWERK INITIALISIERUNG
    # ==========================================
    print("--- Initialisiere Netzwerk ---")
    net = ReLUNetwork(input_dim=d, output_dim=1, width=32, depth=2, init_method="he_normal")

    # Spezielle Initialisierung, damit das Netz im Integrationsbereich viele Falten hat
    init_many_affine_pieces(net)
    net.eval()  # Wichtig: Immer in den Evaluierungsmodus setzen!

    # ==========================================
    # 3. ADAPTIVE L_P NORM BERECHNUNG
    # ==========================================
    lp = AdaptiveLpNorm(d=d, neural_net=net, is_relu=True)

    print(f"\nStarte adaptive L_{int(p)}-Norm Berechnung (m={m}, theta={theta})...")
    start_time = time.time()

    # Unterdrücke die ausufernden Warnungen der Bounding-Bibliothek
    with silence_everything():
        # WICHTIG: eta_type="fval" nutzt die native auto_LiRPA Funktion!
        out = lp.norm(p=p, m=m, theta=theta, refiner=refiner, eta_type="fval")

    duration = time.time() - start_time
    results = lp.out_adap

    # ==========================================
    # 4. MONTE CARLO GROUND TRUTH & AUSGABE
    # ==========================================
    print("\nBerechne Monte Carlo Ground Truth zum Vergleich...")
    mc_truth = monte_carlo_lp_norm(net, d=d, p=p, n_samples=500_000)

    print(f"\n{'=' * 40}")
    print(f"ERGEBNISSE (Dauer: {duration:.2f} Sekunden)")
    print(f"{'=' * 40}")
    print(f"Finale berechnete L_{int(p)} Norm : {results['approx']:.6f}")
    print(f"Finaler a-posteriori Fehler : {results['error']:.6f}")
    print(f"Monte Carlo Ground Truth    : {mc_truth:.6f}")
    print(f"Anzahl der Boxen am Ende    : {results['size']}")
    print(f"{'=' * 40}")
    plot_convergence(results, d)
    plot_theorem_convergence(results, d, theta, rho=0.5)


def plot_convergence(results, d):
    sizes = np.array(results["size_hist"])
    post_err = np.array(results["post_hist"])
    prior_err = np.array(results["prior_hist"])
    iterations = np.arange(len(sizes))

    # --- Plot 1: Konvergenzrate über Anzahl der Boxen (N) ---
    plt.figure(figsize=(10, 8))

    plt.plot(sizes, post_err, '-o', color='#D55E00', linewidth=2.5, markersize=8, label='A-posteriori Error (Adaptive)')
    plt.plot(sizes, prior_err, '--', color='#4A7BC7', linewidth=2, label='A-priori Bound')

    # Referenzlinie (O(N^{-1/d})) für gleichmäßige Verfeinerung
    n0, e0 = sizes[0], post_err[0]
    x_theory = np.linspace(n0, sizes[-1], 100)
    rate = -1.0 / d
    plt.plot(x_theory, e0 * (x_theory / n0) ** rate, 'k--', linewidth=2, label=f'Reference Rate $O(N^{{{rate:.2f}}})$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Grid Boxes ($N$)', fontsize=16)
    plt.ylabel(r'$L_p$ Error Bound', fontsize=16)
    plt.title(f'Adaptive $L_p$ Norm Convergence ($d={d}$)', fontsize=18)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("lp_norm_convergence_boxes.pdf", dpi=300)
    plt.show()

    # --- Plot 2: Geometrischer Abfall über Iterationen ---
    plt.figure(figsize=(10, 8))
    plt.plot(iterations, post_err, '-s', color='#85C57A', linewidth=2.5, markersize=8, label='A-posteriori Error')

    plt.yscale('log')
    plt.xlabel('Adaption Iteration $k$', fontsize=16)
    plt.ylabel(r'Global Error Bound $\eta_k$', fontsize=16)
    plt.title('Geometric Error Decay (Dörfler Marking)', fontsize=18)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("lp_norm_geometric_decay.pdf", dpi=300)
    plt.show()


def plot_theorem_convergence(results, d, theta, rho=0.5):
    """
    Generiert die Konvergenzplots basierend auf dem Theorem.
    rho: Die theoretische lokale Fehlerrate aus deinem Beweis (z.B. 0.5 bei Halbierung).
    """
    sizes = np.array(results["size_hist"])
    post_err = np.array(results["post_hist"])
    prior_err = np.array(results["prior_hist"])
    iterations = np.arange(len(sizes))

    # ==========================================================
    # PLOT 1: Geometrischer Abfall über Iterationen (Dein Theorem)
    # ==========================================================
    plt.figure(figsize=(10, 8))

    # Empirischer Fehler
    plt.plot(iterations, post_err, '-s', color='#85C57A', linewidth=2.5, markersize=8,
             label=r'Empirische Schranke $\tilde{\eta}_n$')

    # Theoretische Rate berechnen: q = 1 - theta * (1 - rho)
    q = 1.0 - theta * (1.0 - rho)
    eta_0 = post_err[0]
    theory_decay = eta_0 * (q ** iterations)

    # Theoretische Kurve einzeichnen
    plt.plot(iterations, theory_decay, '--k', linewidth=2.5,
             label=fr'Theorem Rate $q^n \tilde{{\eta}}_0$ ($q={q:.3f}$)')

    plt.yscale('log')
    plt.xlabel('Adaptions-Iteration $n$', fontsize=16)
    plt.ylabel(r'Globaler Fehler-Estimator $\tilde{\eta}_n$', fontsize=16)
    plt.title('Geometrischer Abfall gemäß Theorem', fontsize=18)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("lp_norm_theorem_decay.pdf", dpi=300)
    print("Gespeichert: lp_norm_theorem_decay.pdf")

    # ==========================================================
    # PLOT 2: Konvergenzrate über Anzahl der Boxen (N)
    # ==========================================================
    plt.figure(figsize=(10, 8))

    plt.plot(sizes, post_err, '-o', color='#D55E00', linewidth=2.5, markersize=8,
             label=r'A-posteriori Error $\tilde{\eta}_n$')
    plt.plot(sizes, prior_err, '--', color='#4A7BC7', linewidth=2, label='A-priori Bound')

    # Referenzlinie für Standard-Konvergenz O(N^{-1/d})
    n0, e0 = sizes[0], post_err[0]
    x_theory = np.linspace(n0, sizes[-1], 100)
    rate = -1.0 / d
    plt.plot(x_theory, e0 * (x_theory / n0) ** rate, 'k--', linewidth=2,
             label=f'Reference Rate $\mathcal{{O}}(N^{{{rate:.2f}}})$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Anzahl der Boxen ($N$)', fontsize=16)
    plt.ylabel(r'$L_p$ Fehlerschranke', fontsize=16)
    plt.title(f'Adaptive $L_p$ Norm Konvergenz ($d={d}$)', fontsize=18)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("lp_norm_convergence_boxes.pdf", dpi=300)
    print("Gespeichert: lp_norm_convergence_boxes.pdf")
if __name__ == "__main__":
    main()