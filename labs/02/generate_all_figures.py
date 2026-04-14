"""Generate all report figures for Lab 02.

Run from /home/ikaros/projects/ay-128/labs/02/ as the working directory.
"""
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).parent / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

successes = []
failures = []

def attempt(name, fn):
    """Run fn(); log success or failure."""
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"{'='*60}")
    try:
        fn()
        successes.append(name)
        print(f"  -> SUCCESS: {name}")
    except Exception:
        traceback.print_exc()
        failures.append(name)
        print(f"  -> FAILED: {name}")
    finally:
        plt.close("all")

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
print("Loading data files...")

spec_data = np.load("training_spectra.npz", allow_pickle=True)
wavelength = spec_data["wavelength"]
flux = spec_data["flux"]
error = spec_data["error"]
labels = spec_data["labels"]
label_names_raw = list(spec_data["label_names"])
apogee_ids = spec_data["apogee_ids"]
continuum_mask = spec_data["continuum_mask"].astype(bool)
fields = spec_data["fields"]

model_data = np.load("cannon_model.npz", allow_pickle=True)
train_idx = model_data["train_idx"]
cv_idx = model_data["cv_idx"]

cv_data = np.load("cv_results.npz", allow_pickle=True)
nn_data = np.load("nn_results.npz", allow_pickle=True)

from ugdatalab.methods.cannon import CannonModel
from ugdatalab.models.apogee.constants import LABEL_NAMES, LABEL_LATEX

model = CannonModel(
    theta=model_data["theta"],
    scatter=model_data["scatter"],
    label_names=list(model_data["label_names"]),
    label_means=model_data["label_means"],
    label_stds=model_data["label_stds"],
    wavelength=model_data["wavelength"],
    chi2_r=float(model_data["chi2_r"]),
)

import plotters

print("Data loaded successfully.")

# ---------------------------------------------------------------------------
# 1. Label corner plot
# ---------------------------------------------------------------------------
def gen_label_corner():
    plotters.plot_label_corner(labels, LABEL_LATEX)

# ---------------------------------------------------------------------------
# 2. Label corner by field (train vs CV)
# ---------------------------------------------------------------------------
def gen_label_corner_by_field():
    labels_train = labels[train_idx]
    labels_cv_split = labels[cv_idx]
    split_labels = np.concatenate([labels_train, labels_cv_split], axis=0)
    split_names = np.array(
        ["Train"] * len(train_idx) + ["CV"] * len(cv_idx)
    )
    plotters.plot_label_corner_by_field(split_labels, LABEL_LATEX, split_names)

# ---------------------------------------------------------------------------
# 3. Example raw spectrum (needs APOGEE download)
# ---------------------------------------------------------------------------
def gen_example_spectrum():
    from ugdatalab.models.apogee.spectra import _get_apstar_spectra
    example_id = "2M21235315+1244123"
    # Find telescope and field from training data
    ids_str = np.array(apogee_ids, dtype=str)
    idx = np.where(ids_str == example_id)[0][0]
    field = str(fields[idx])
    # We need to query APOGEE for telescope info.  The training_spectra.npz
    # doesn't store telescope, so we try known values for this star.
    # From the notebook: it's in M15, telescope=apo25m
    spec_raw = _get_apstar_spectra(example_id, "apo25m", "M15")
    plotters.plot_example_spectrum(
        spec_raw["wavelength"], spec_raw["flux"], spec_raw["error"],
        apogee_id=example_id,
    )
    return spec_raw  # reuse for bitmask/normalization

# We'll store spec_raw for reuse
_spec_raw_cache = {}

def gen_example_spectrum_cached():
    spec_raw = gen_example_spectrum()
    _spec_raw_cache["spec_raw"] = spec_raw

# ---------------------------------------------------------------------------
# 4. Bitmask diagnostic
# ---------------------------------------------------------------------------
def gen_bitmask_diagnostic():
    if "spec_raw" not in _spec_raw_cache:
        from ugdatalab.models.apogee.spectra import _get_apstar_spectra
        spec_raw = _get_apstar_spectra("2M21235315+1244123", "apo25m", "M15")
        _spec_raw_cache["spec_raw"] = spec_raw
    spec_raw = _spec_raw_cache["spec_raw"]
    plotters.plot_bitmask_diagnostic(
        spec_raw["wavelength"], spec_raw["flux"], spec_raw["error"],
        spec_raw["bitmask"], apogee_id="2M21235315+1244123",
    )

# ---------------------------------------------------------------------------
# 5. Normalization diagnostic
# ---------------------------------------------------------------------------
def gen_normalization():
    if "spec_raw" not in _spec_raw_cache:
        from ugdatalab.models.apogee.spectra import _get_apstar_spectra
        spec_raw = _get_apstar_spectra("2M21235315+1244123", "apo25m", "M15")
        _spec_raw_cache["spec_raw"] = spec_raw
    spec_raw = _spec_raw_cache["spec_raw"]

    from ugdatalab.models.apogee.spectra import _apply_bitmask, _normalize_spectrum

    continuum_data = np.load("continuum_wavelengths.npz")
    cm = continuum_data["continuum"].astype(bool)
    wl = continuum_data["wavelengths"]

    flux_masked, error_masked = _apply_bitmask(
        spec_raw["flux"], spec_raw["error"], spec_raw["bitmask"],
    )
    flux_norm, error_norm, continuum_fit = _normalize_spectrum(
        flux_masked, error_masked, wl, cm,
        spec_raw["chips"], degree=4,
    )
    plotters.plot_normalization_diagnostic(
        wl, flux_masked, error_masked, continuum_fit, flux_norm, error_norm,
        cm, apogee_id="2M21235315+1244123",
    )

# ---------------------------------------------------------------------------
# 6. Bitmask frequency (needs all raw spectra - expensive; skip if no cache)
# ---------------------------------------------------------------------------
def gen_bitmask_frequency():
    """This requires downloading ALL training spectra raw data.
    We'll try to use the training_spectra.npz flux as a proxy,
    but the real bitmask_frequency needs raw flux and bitmasks.
    Skip if we can't easily get them.
    """
    # The bitmask frequency plot requires raw (un-normalized) flux and bitmasks
    # for ALL training stars. This isn't stored in training_spectra.npz.
    # We skip this one as it would require downloading ~1900 spectra.
    raise RuntimeError(
        "Bitmask frequency requires raw spectra for all ~1886 stars "
        "(not stored in training_spectra.npz). Skipping."
    )

# ---------------------------------------------------------------------------
# 7. Similar stars comparison
# ---------------------------------------------------------------------------
def gen_similar_stars():
    example_id = "2M21235315+1244123"
    ids_str = np.array(apogee_ids, dtype=str)
    ref_idx = int(np.where(ids_str == example_id)[0][0])
    plotters.plot_similar_stars_comparison(
        wavelength, flux, error, labels, apogee_ids,
        ref_idx=ref_idx, n_similar=5,
    )

# ---------------------------------------------------------------------------
# 8. Training prediction
# ---------------------------------------------------------------------------
def gen_training_prediction():
    target_id = "2M03533659+2512012"
    all_ids = np.array(apogee_ids, dtype=str)
    match = np.where(all_ids == target_id)[0]
    if len(match) == 0:
        raise ValueError(f"{target_id} not found")
    full_idx = int(match[0])
    flux_obs = flux[full_idx]
    error_obs = error[full_idx]
    flux_pred = model.predict(labels[full_idx])
    plotters.plot_training_prediction(
        wavelength, flux_obs, error_obs, flux_pred,
        apogee_id=target_id, wl_min=16000, wl_max=16100,
    )

# ---------------------------------------------------------------------------
# 9. Gradient spectra
# ---------------------------------------------------------------------------
def gen_gradient_spectra():
    plotters.plot_gradient_spectra(model)

# ---------------------------------------------------------------------------
# 10. Scatter spectrum
# ---------------------------------------------------------------------------
def gen_scatter_spectrum():
    plotters.plot_scatter_spectrum(model)

# ---------------------------------------------------------------------------
# 11. Label recovery (CV)
# ---------------------------------------------------------------------------
def gen_label_recovery():
    plotters.plot_label_recovery(
        cv_data["true_labels"], cv_data["fitted_labels"], LABEL_LATEX,
    )

# ---------------------------------------------------------------------------
# 12. Kiel diagram with isochrones
# ---------------------------------------------------------------------------
def gen_kiel_diagram():
    from ugdatalab.models.isochrones import _get_mist_isochrone
    iso_solar = _get_mist_isochrone(6.0, 0.0)
    iso_metal_poor = _get_mist_isochrone(6.0, -1.0)
    isochrone_tracks = [
        (r"MIST $[\mathrm{Fe/H}]=0$", iso_solar),
        (r"MIST $[\mathrm{Fe/H}]=-1$", iso_metal_poor),
    ]
    fitted_labels_cv = cv_data["fitted_labels"]
    ax = plotters.plot_kiel_diagram(fitted_labels_cv, isochrone_tracks)
    # plot_kiel_diagram already calls savefig internally

# ---------------------------------------------------------------------------
# 13. MCMC corner plot (mystery spectrum)
# ---------------------------------------------------------------------------
def gen_methods_corner():
    from ugdatalab.models.apogee import Spectrum
    from ugdatalab.methods.cannon_likelihood import CannonLabelLikelihood
    from ugdatalab.methods.bayesian.mcmc import nuts_sample
    from ugdatalab.plotters.bayesian import plot_corner

    mystery_path = Path("../../course_materials_sp2026/labs/lab_2/mystery_spec_wiped.fits")
    continuum_path = Path("continuum_wavelengths.npz")
    mystery = Spectrum(mystery_path, continuum_path)
    flux_norm = mystery.flux[0]
    error_norm = mystery.error[0]

    lk = CannonLabelLikelihood(
        x=model.wavelength,
        y=flux_norm,
        y_err=error_norm,
        model=model,
    )
    result = nuts_sample(lk, n_steps=2000, n_burn=1000, seed=42)
    print(f"  MCMC chi2_r: {result.chi2_r:.3f}")
    print(f"  Posterior median labels: {result.theta}")

    fig = plot_corner(result)
    plotters.savefig(fig, "fig_methods_corner.pdf")

    # Store result for mystery prediction plot
    _spec_raw_cache["mcmc_result"] = result
    _spec_raw_cache["mystery_flux"] = flux_norm
    _spec_raw_cache["mystery_error"] = error_norm

# ---------------------------------------------------------------------------
# 14. Mystery spectrum prediction
# ---------------------------------------------------------------------------
def gen_mystery_prediction():
    if "mcmc_result" not in _spec_raw_cache:
        # Run MCMC if not already done
        gen_methods_corner()

    result = _spec_raw_cache["mcmc_result"]
    flux_norm = _spec_raw_cache["mystery_flux"]
    error_norm = _spec_raw_cache["mystery_error"]

    flux_pred_mcmc = model.predict(result.theta)
    plotters.plot_mystery_prediction(
        model.wavelength, flux_norm, error_norm, flux_pred_mcmc,
        scatter=model.scatter, wl_min=15200, wl_max=16900,
    )

# ---------------------------------------------------------------------------
# 15. Metallicity sequence
# ---------------------------------------------------------------------------
def gen_metallicity_sequence():
    feh_values = np.arange(-1.0, 0.75, 0.25)
    plotters.plot_metallicity_sequence(
        model, feh_values,
        teff=4800, logg=2.5, mg_fe=0.0, si_fe=0.0,
        wl_min=16000, wl_max=16200,
    )

# ---------------------------------------------------------------------------
# 16. RGB evolution
# ---------------------------------------------------------------------------
def gen_rgb_evolution():
    from ugdatalab.models.isochrones import _get_mist_isochrone
    iso_solar = _get_mist_isochrone(6.0, 0.0)
    rgb_mask = (
        (iso_solar["logg"] >= 0.5)
        & (iso_solar["logg"] <= 3.5)
        & (iso_solar["Teff"] < 5500)
    )
    rgb = iso_solar[rgb_mask].sort_values("logg", ascending=False)
    n_points = min(7, len(rgb))
    step = max(1, len(rgb) // n_points)
    rgb_sample = rgb.iloc[::step][:n_points]

    plotters.plot_rgb_evolution(
        model, rgb_sample["Teff"].values, rgb_sample["logg"].values,
        feh=0.0, mg_fe=0.0, si_fe=0.0,
        wl_min=16000, wl_max=16200,
    )

# ---------------------------------------------------------------------------
# 17. NN loss curves (requires re-training)
# ---------------------------------------------------------------------------
def gen_nn_loss():
    import torch
    import torch.nn as tnn
    from torch.utils.data import DataLoader, TensorDataset

    # Replicate NN training to get loss curves
    flux_train = flux[train_idx]
    labels_train = labels[train_idx]
    flux_cv_split = flux[cv_idx]
    labels_cv_split = labels[cv_idx]

    label_mean = np.mean(labels_train, axis=0)
    label_std = np.std(labels_train, axis=0)
    labels_train_norm = (labels_train - label_mean) / label_std
    labels_cv_norm = (labels_cv_split - label_mean) / label_std

    flux_train_clean = np.nan_to_num(flux_train, nan=0.0, posinf=0.0, neginf=0.0)
    flux_cv_clean = np.nan_to_num(flux_cv_split, nan=0.0, posinf=0.0, neginf=0.0)

    device = torch.device("cpu")
    train_dataset = TensorDataset(
        torch.tensor(flux_train_clean, dtype=torch.float32),
        torch.tensor(labels_train_norm, dtype=torch.float32),
    )
    cv_dataset = TensorDataset(
        torch.tensor(flux_cv_clean, dtype=torch.float32),
        torch.tensor(labels_cv_norm, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=len(cv_idx), shuffle=False)

    n_pixels = flux.shape[1]
    n_labels_count = labels.shape[1]

    torch.manual_seed(42)
    net = tnn.Sequential(
        tnn.Linear(n_pixels, 512),
        tnn.ReLU(),
        tnn.Linear(512, 256),
        tnn.ReLU(),
        tnn.Linear(256, n_labels_count),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = tnn.MSELoss()

    n_epochs = 200
    patience = 20
    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_state = None

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        net.eval()
        with torch.no_grad():
            for X_val, y_val in cv_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_loss = criterion(net(X_val), y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    plotters.plot_nn_loss(train_losses, val_losses)

# ---------------------------------------------------------------------------
# 18. NN label recovery
# ---------------------------------------------------------------------------
def gen_nn_label_recovery():
    plotters.plot_nn_label_recovery(
        nn_data["true_labels"], nn_data["nn_fitted_labels"], LABEL_LATEX,
    )


# ===========================================================================
# Run all figure generators
# ===========================================================================
if __name__ == "__main__":
    # First: download raw spectrum (reused by 3 figures)
    attempt("fig_example_spectrum.pdf", gen_example_spectrum_cached)
    attempt("fig_bitmask_diagnostic.pdf", gen_bitmask_diagnostic)
    attempt("fig_normalization.pdf", gen_normalization)

    # Figures from cached NPZ data (no downloads needed)
    attempt("fig_label_corner.pdf", gen_label_corner)
    attempt("fig_label_corner_by_field.pdf", gen_label_corner_by_field)
    attempt("fig_similar_stars.pdf", gen_similar_stars)
    attempt("fig_training_prediction.pdf", gen_training_prediction)
    attempt("fig_gradient_spectra.pdf", gen_gradient_spectra)
    attempt("fig_scatter_spectrum.pdf", gen_scatter_spectrum)
    attempt("fig_label_recovery.pdf", gen_label_recovery)
    attempt("fig_kiel_diagram.pdf", gen_kiel_diagram)
    attempt("fig_metallicity_sequence.pdf", gen_metallicity_sequence)
    attempt("fig_rgb_evolution.pdf", gen_rgb_evolution)
    attempt("fig_nn_label_recovery.pdf", gen_nn_label_recovery)

    # NN loss (requires re-training the network)
    attempt("fig_nn_loss.pdf", gen_nn_loss)

    # MCMC-based figures (most expensive - run last)
    attempt("fig_methods_corner.pdf", gen_methods_corner)
    attempt("fig_mystery_prediction.pdf", gen_mystery_prediction)

    # Bitmask frequency (skipped - requires raw spectra download)
    attempt("fig_bitmask_frequency.pdf", gen_bitmask_frequency)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Succeeded: {len(successes)}/{len(successes)+len(failures)}")
    for s in successes:
        print(f"  OK  {s}")
    if failures:
        print(f"\nFailed: {len(failures)}/{len(successes)+len(failures)}")
        for f in failures:
            print(f"  FAIL  {f}")

    # List generated files
    print(f"\nGenerated files in {FIGURES_DIR}:")
    for p in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {p.name}")
