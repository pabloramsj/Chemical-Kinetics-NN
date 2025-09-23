# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 21:37:33 2025

@author: Pablo Ramos
"""

import cantera as ct
import numpy as np
import h5py
import time
from tqdm import tqdm

start = time.time() 

# ===============================
# Configuración
# ===============================
mechanism = "ESH2.yaml"   # mecanismo Cantera
fuel = "H2"
n_conditions = 10


# --- Condiciones iniciales conocidas ---
T0_range = (1000.0, 3000.0)   # K
p0_range = (1e5, 30e5)       # Pa
phi_range = (0.6, 1.6)

# Integración / eventos
t_end   = 2e-4     # s
dt_max  = 1e-8    # s (paso "normal")
min_dt  = 1e-12    # s (límite inferior por seguridad)

# Umbrales de evento (disparan el refinamiento temporal)
threshold_dTdt  = 1e7      # K/s
threshold_wdot  = 1e3      # kmol/m^3/s

# Parámetros de refinamiento por evento
event_refine_factor = 15   # reduce dt a dt_max / factor
event_burst_steps   = 10   # nº de pasos finos tras disparar evento

#Umbrales por debajo
low_dTdt = 1e5
low_wdot = 10
low_qrel = 1e-4
quiet_steps = 5
dt_coarse_max = 1e-6

# Criterio de parada por "equilibrio" relativo
tol_qrel       = 1e-6
tol_wrel       = 1e-6
stable_needed  = 15

out_path = "Prueba_Sampling.h5"

# ===============================
# Funciones auxiliares
# ===============================

def equivalence_composition(phi, fuel="H2"):
    # Mezcla H2/air estequiométrica base: H2 + 0.5 O2 (+ 1.88 N2)
    O2 = 1.0
    N2 = 3.76
    F_stoich = 0.5 * O2
    F = phi * F_stoich
    comp = {fuel: F, "O2": O2, "N2": N2}
    tot = sum(comp.values())
    for k in comp:
        comp[k] /= tot
    return comp

def qdot_Wm3(gas: ct.Solution):
    hk = gas.partial_molar_enthalpies
    wdot = gas.net_production_rates
    return -np.dot(hk, wdot)  # Potencia volumétrica liberada (+) o absorbida (-)

def integrate_condition(
    gas,
    T0, p0, phi,
    case_id=0,
    dt_max=1e-10,
    min_dt=1e-12,
    threshold_dTdt=1e7,
    threshold_wdot=1e3,
    event_refine_factor=20,
    event_burst_steps=80,
    t_end=2e-4,
    tol_qrel=1e-6,
    tol_wrel=1e-6,
    stable_needed=15
):
    """
    Integra un caso único con refinamiento temporal alrededor de eventos:
    si |dT/dt| o max|wdot| superan umbrales, se reduce temporalmente el paso.

    El criterio de parada combina:
      - tiempo máximo t_end
      - relajación relativa de |qdot| y max|wdot|
    """
    comp = equivalence_composition(phi, fuel)
    gas.TPX = T0, p0, comp

    r = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([r])
    net.rtol = 1e-10
    net.atol = 1e-20

    # Históricos
    times, T_hist, p_hist = [], [], []
    Y_hist, wdot_hist, qdot_hist, ids = [], [], [], []

    t = 0.0
    current_dt = float(dt_max)
    burst_left = 0

    last_T, last_time = gas.T, 0.0

    # Picos (para criterios relativos de parada)
    q_peak = 0.0
    wdot_peak = 0.0
    stable_steps = 0

    while True:
        # Avanza con el paso actual (posiblemente refinado)
        t_next = t + current_dt
        net.advance(t_next)
        t = t_next

        # Guardar estado
        times.append(t)
        T_hist.append(gas.T)
        p_hist.append(gas.P)
        Y_hist.append(gas.Y.copy())
        wdot = gas.net_production_rates.copy()
        wdot_hist.append(wdot)
        qdot_val = qdot_Wm3(gas)
        qdot_hist.append(qdot_val)
        ids.append(case_id)

        # Actualizar máximos para referencias relativas
        q_peak = max(q_peak, abs(qdot_val))
        wdot_peak = max(wdot_peak, float(np.max(np.abs(wdot))))

        # --- Detección de evento (igual a tu idea, pero con refinamiento real) ---
        dTdt = (gas.T - last_T) / (t - last_time + 1e-30)
        event = (abs(dTdt) > threshold_dTdt) or (float(np.max(np.abs(wdot))) > threshold_wdot)

        if event:
            # Entra/bombea modo "burst": pasos más pequeños durante event_burst_steps
            burst_left = max(burst_left, event_burst_steps)
            current_dt = max(dt_max / event_refine_factor, min_dt)
        else:
            if burst_left > 0:
                burst_left -= 1
                if burst_left == 0:
                    current_dt = dt_max  # volver a paso normal

        last_T, last_time = gas.T, t
        
        
        # --- Coarsening (aumentar dt cuando la cinética está “quieta”) ---
        # Contador de pasos tranquilos
        if 'quiet_counter' not in locals():
            quiet_counter = 0
        
        is_quiet = (
            abs(dTdt) < low_dTdt and
            float(np.max(np.abs(wdot))) < low_wdot and
            (q_peak > 0.0 and abs(qdot_val)/(q_peak + 1e-30) < low_qrel)
        )
        
        # Salvaguardas por cambio por paso
        delta_T_ok = abs(gas.T - last_T) <= 2.0  # ΔT_max_per_step
        deltaY = np.abs(gas.Y - Y_hist[-1]) if len(Y_hist) else np.zeros_like(gas.Y)
        fracY_ok = (deltaY <= 0.02 * np.maximum(1e-12, np.abs(gas.Y))) .all()
        
        if is_quiet and delta_T_ok and fracY_ok and burst_left == 0:
            quiet_counter += 1
        else:
            quiet_counter = 0
        
        if quiet_counter >= quiet_steps:
            # Suavemente hacia arriba, con techo
            current_dt = min(current_dt * 2.0, dt_coarse_max)  # g_coarsen
            quiet_counter = 0  # reinicia para evitar escaladas rápidas


        # ---- Criterios de parada ----
        if t >= t_end:
            break

        if q_peak > 0.0 and wdot_peak > 0.0:
            rel_q = abs(qdot_val) / (q_peak + 1e-30)
            rel_w = float(np.max(np.abs(wdot))) / (wdot_peak + 1e-30)

            if (rel_q < tol_qrel) and (rel_w < tol_wrel):
                stable_steps += 1
            else:
                stable_steps = 0

            if stable_steps >= stable_needed:
                break

    # Deduplicar timestamps por si caen iguales numéricamente
    times = np.asarray(times)
    T_hist = np.asarray(T_hist)
    p_hist = np.asarray(p_hist)
    Y_hist = np.asarray(Y_hist)
    wdot_hist = np.asarray(wdot_hist)
    qdot_hist = np.asarray(qdot_hist)
    ids = np.asarray(ids)

    unique_idx = np.unique(times, return_index=True)[1]
    unique_idx.sort()

    return (times[unique_idx],
            T_hist[unique_idx],
            p_hist[unique_idx],
            Y_hist[unique_idx],
            wdot_hist[unique_idx],
            qdot_hist[unique_idx],
            ids[unique_idx])

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    gas = ct.Solution(mechanism)

    with h5py.File(out_path, "w") as f:
        # Datasets principales
        f.create_dataset("states/time", (0,), maxshape=(None,), dtype="f8", chunks=True)
        f.create_dataset("states/T", (0,), maxshape=(None,), dtype="f8", chunks=True)
        f.create_dataset("states/p", (0,), maxshape=(None,), dtype="f8", chunks=True)
        f.create_dataset("states/Y", (0, gas.n_species), maxshape=(None, gas.n_species), dtype="f8", chunks=True)
        f.create_dataset("states/case_id", (0,), maxshape=(None,), dtype="i4", chunks=True)
        f.create_dataset("targets/wdot", (0, gas.n_species), maxshape=(None, gas.n_species), dtype="f8", chunks=True)
        f.create_dataset("targets/qdot", (0,), maxshape=(None,), dtype="f8", chunks=True)

        # Metadatos
        f.create_dataset("meta/species", data=np.array([s.encode("utf8") for s in gas.species_names], dtype="S"))
        f.attrs["mechanism"] = mechanism

        # Grupo con condiciones iniciales
        meta_ic = f.create_group("meta/initial_conditions")
        meta_ic.create_dataset("T0", (n_conditions,), dtype="f8")
        meta_ic.create_dataset("p0", (n_conditions,), dtype="f8")
        meta_ic.create_dataset("phi", (n_conditions,), dtype="f8")
        meta_ic.create_dataset("index_start", (n_conditions,), dtype="i8")
        meta_ic.create_dataset("index_end", (n_conditions,), dtype="i8")

        rng = np.random.default_rng(42)
        for i in tqdm(range(n_conditions), desc="Simulating conditions"):
            T0 = rng.uniform(*T0_range)
            p0 = rng.uniform(*p0_range)
            phi = rng.uniform(*phi_range)

            times, T_hist, p_hist, Y_hist, wdot_hist, qdot_hist, ids = integrate_condition(gas, T0, p0, phi, i)

            def append(ds_name, arr):
                ds = f[ds_name]
                n_old = ds.shape[0]
                n_new = n_old + arr.shape[0]
                ds.resize((n_new,) + ds.shape[1:])
                ds[n_old:n_new, ...] = arr

            start_idx = f["states/time"].shape[0]
            append("states/time", times)
            append("states/T", T_hist)
            append("states/p", p_hist)
            append("states/Y", Y_hist)
            append("states/case_id", ids)
            append("targets/wdot", wdot_hist)
            append("targets/qdot", qdot_hist)
            end_idx = f["states/time"].shape[0]

            # Guardar condiciones iniciales y rangos de índices
            meta_ic["T0"][i] = T0
            meta_ic["p0"][i] = p0
            meta_ic["phi"][i] = phi
            meta_ic["index_start"][i] = start_idx
            meta_ic["index_end"][i] = end_idx

    print(f"Dataset escrito en: {out_path}")

end = time.time()     # Marca final
print(f"Tiempo de ejecución: {end - start:.3f} segundos")