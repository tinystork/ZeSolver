#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """


import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

try:
    from astropy.io import fits
except Exception as exc:
    print("This tool requires astropy. Install with: pip install astropy")
    raise

WCS_PREFIXES = (
    "CRVAL", "CRPIX", "CD", "PC", "CDELT", "CTYPE", "CROTA",
    "PV", "A_", "B_", "AP_", "BP_",  # SIP coefficient cards like A_0_2
)
WCS_SINGLE_KEYS = {
    "WCSAXES", "LONPOLE", "LATPOLE", "EQUINOX", "RADESYS", "RADECSYS",
    "CUNIT", "CUNIT1", "CUNIT2", "CUNIT3", "CUNIT4",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",  # SIP orders
    "A_DMAX", "B_DMAX", "AP_DMAX", "BP_DMAX",
}


REFERENCE_TOKENS = ("ASTAP", "ZEMOSAIC")


def _normalize_comment_text(value) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split()).lower()


def _contains_solver_ref(value) -> bool:
    text = "" if value is None else str(value)
    upper = text.upper()
    return any(tok in upper for tok in REFERENCE_TOKENS)


def clean_non_wcs_metadata_inplace(hdr) -> int:
    """Drop redundant COMMENT/HISTORY cards and solver reference traces."""
    to_delete_idx = []
    seen_comment = set()
    seen_history = set()

    for idx, card in enumerate(hdr.cards):
        key = str(card.keyword).upper()

        if key in ("COMMENT", "HISTORY"):
            norm = _normalize_comment_text(card.value)
            if _contains_solver_ref(norm):
                to_delete_idx.append(idx)
                continue
            if not norm:
                continue
            if key == "COMMENT":
                if norm in seen_comment:
                    to_delete_idx.append(idx)
                else:
                    seen_comment.add(norm)
            else:
                if norm in seen_history:
                    to_delete_idx.append(idx)
                else:
                    seen_history.add(norm)
            continue

                # Drop explicit solver provenance keys/comments
        if _contains_solver_ref(key):
            to_delete_idx.append(idx)
            continue
        if _contains_solver_ref(card.comment):
            to_delete_idx.append(idx)
            continue

    deleted = 0
    for idx in reversed(to_delete_idx):
        try:
            del hdr[idx]
            deleted += 1
        except Exception:
            pass
    return deleted


def header_has_wcs(hdr) -> bool:
    """Conservatively decide if a header 'has WCS'."""
    keys = set(hdr.keys())
    # Common telltales:
    telltales = {"WCSAXES", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"}
    if keys.intersection(telltales):
        return True
    # Any matrix-like or SIP/pv terms?
    for k in keys:
        uk = str(k).upper()
        if uk.startswith(WCS_PREFIXES) or uk in WCS_SINGLE_KEYS:
            return True
    return False

def clean_wcs_header_inplace(hdr) -> int:
    """Remove WCS cards + redundant/commentary solver traces. Returns deletions."""
    to_delete = []
    for key in list(hdr.keys()):
        uk = str(key).upper()
        if uk in WCS_SINGLE_KEYS:
            to_delete.append(key)
            continue
        if uk.startswith(WCS_PREFIXES):
            to_delete.append(key)
            continue
        # Generic CUNITi (any axis)
        if uk.startswith("CUNIT"):
            to_delete.append(key)

    deleted_wcs = 0
    for k in to_delete:
        try:
            del hdr[k]
            deleted_wcs += 1
        except Exception:
            pass

    deleted_meta = clean_non_wcs_metadata_inplace(hdr)
    deleted = deleted_wcs + deleted_meta

    if deleted:
        try:
            hdr.add_history(
                f"[WCS-CLEAN] Removed {deleted_wcs} WCS cards + {deleted_meta} metadata cards on {datetime.utcnow().isoformat()}Z"
            )
        except Exception:
            pass
    return deleted

def process_fits(path: str, *, dry_run: bool, backup: bool, only_if_wcs: bool, all_hdus: bool) -> tuple[int, int]:
    """
    Returns (deleted_cards_total, edited_hdus_count).
    """
    deleted_total = 0
    edited_hdus = 0

    # Pre-check: skip non-files
    if not os.path.isfile(path):
        return (0, 0)

    with fits.open(path, mode="update") as hdul:
        targets = range(len(hdul)) if all_hdus else [0]
        # Optional 'only if WCS' gate: check any target has WCS
        if only_if_wcs:
            ok = any(header_has_wcs(hdul[i].header) for i in targets)
            if not ok:
                return (0, 0)

        if not dry_run and backup:
            try:
                bak = path + ".bak"
                if not os.path.exists(bak):
                    hdul.writeto(bak, overwrite=False, output_verify="fix")
            except Exception:
                # Backup is best-effort; continue anyway
                pass

        for i in targets:
            hdr = hdul[i].header
            before = len(hdr)
            n = clean_wcs_header_inplace(hdr)
            if n > 0:
                deleted_total += n
                edited_hdus += 1

        if not dry_run and deleted_total > 0:
            hdul.flush()

    return (deleted_total, edited_hdus)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FITS WCS Cleaner")
        try:
            self.iconbitmap(default="")  # ignore if not found
        except Exception:
            pass
        self.geometry("900x600")
        self.minsize(800, 520)

        self.paths = []  # files selected
        self.var_dry = tk.BooleanVar(value=True)
        self.var_backup = tk.BooleanVar(value=True)
        self.var_only_wcs = tk.BooleanVar(value=True)
        self.var_all_hdus = tk.BooleanVar(value=False)
        self.var_recursive = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Add FITS files...", command=self.add_files).pack(side="left")
        ttk.Button(top, text="Add folder...", command=self.add_folder).pack(side="left", padx=(8,0))
        ttk.Button(top, text="Clear list", command=self.clear_list).pack(side="left", padx=(8,0))

        opts = ttk.Frame(self, padding=(8,4))
        opts.pack(fill="x")
        ttk.Checkbutton(opts, text="Dry-run (don’t modify files)", variable=self.var_dry).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Create .bak backups", variable=self.var_backup).grid(row=0, column=1, sticky="w", padx=(16,0))
        ttk.Checkbutton(opts, text="Process only if WCS present", variable=self.var_only_wcs).grid(row=0, column=2, sticky="w", padx=(16,0))
        ttk.Checkbutton(opts, text="Clean all HDUs (not just PRIMARY)", variable=self.var_all_hdus).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Recurse into subfolders", variable=self.var_recursive).grid(row=1, column=1, sticky="w", padx=(16,0))

        mid = ttk.Frame(self, padding=(8,4))
        mid.pack(fill="both", expand=True)
        cols = ("file","status","deleted","hdus")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (520, 120, 80, 60)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="w" if c=="file" else "center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", expand=True, side="left")
        ttk.Button(bottom, text="Scan selection", command=self.scan_selection).pack(side="left", padx=(8,0))
        ttk.Button(bottom, text="Clean selection", command=self.clean_selection).pack(side="left", padx=(8,0))

        logf = ttk.LabelFrame(self, text="Log", padding=6)
        logf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.txt = tk.Text(logf, height=8, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self._log("Ready.")

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select FITS files",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
        )
        if not paths: return
        for p in paths:
            if p not in self.paths:
                self.paths.append(p)
                self.tree.insert("", "end", values=(p, "", "", ""))

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder: return
        count = 0
        for root, dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith((".fits",".fit",".fts")):
                    p = os.path.join(root, fn)
                    if p not in self.paths:
                        self.paths.append(p)
                        self.tree.insert("", "end", values=(p, "", "", ""))
                        count += 1
            if not self.var_recursive.get():
                break
        self._log(f"Added {count} file(s) from folder.")

    def clear_list(self):
        self.paths.clear()
        for it in self.tree.get_children():
            self.tree.delete(it)
        self._log("Cleared list.")

    def _iter_selected_rows(self):
        sel = self.tree.selection()
        if not sel:
            sel = self.tree.get_children()
        for iid in sel:
            vals = self.tree.item(iid, "values")
            if not vals: continue
            yield iid, vals[0]

    def scan_selection(self):
        # Update status with YES/NO for WCS presence
        targets = list(self._iter_selected_rows())
        if not targets:
            messagebox.showinfo("Scan", "Nothing to scan.")
            return
        self.progress.configure(maximum=len(targets), value=0)
        hits = 0
        for i, (iid, path) in enumerate(targets, 1):
            try:
                with fits.open(path, memmap=False) as hdul:
                    targets_hdus = range(len(hdul)) if self.var_all_hdus.get() else [0]
                    has = any(header_has_wcs(hdul[h].header) for h in targets_hdus)
                status = "WCS: YES" if has else "WCS: NO"
                if has: hits += 1
                self.tree.item(iid, values=(path, status, "", ""))
            except Exception as exc:
                self.tree.item(iid, values=(path, "ERROR", "", ""))
                self._log(f"[SCAN][ERROR] {path}: {exc}")
            self.progress['value'] = i
            self.update_idletasks()
        self._log(f"Scan done. {hits} / {len(targets)} with WCS.")

    def clean_selection(self):
        targets = list(self._iter_selected_rows())
        if not targets:
            messagebox.showinfo("Clean", "Nothing to clean.")
            return

        dry = self.var_dry.get()
        backup = self.var_backup.get()
        only_if_wcs = self.var_only_wcs.get()
        all_hdus = self.var_all_hdus.get()

        if not dry:
            if not messagebox.askyesno("Confirm", "This will modify files. Proceed?"):
                return

        self.progress.configure(maximum=len(targets), value=0)
        total_deleted = 0
        edited_files = 0

        for i, (iid, path) in enumerate(targets, 1):
            try:
                deleted, edited_hdus = process_fits(
                    path,
                    dry_run=dry,
                    backup=backup,
                    only_if_wcs=only_if_wcs,
                    all_hdus=all_hdus,
                )
                total_deleted += deleted
                if edited_hdus > 0:
                    edited_files += 1
                    status = "CLEANED" if not dry else "DRY"
                else:
                    status = "NO-OP"
                self.tree.item(iid, values=(path, status, str(deleted), str(edited_hdus)))
            except Exception as exc:
                self.tree.item(iid, values=(path, "ERROR", "", ""))
                self._log(f"[CLEAN][ERROR] {path}: {exc}")
                traceback.print_exc()
            self.progress['value'] = i
            self.update_idletasks()

        mode = "Dry-run" if dry else "Applied"
        self._log(f"{mode}: removed {total_deleted} card(s) across {edited_files} file(s).")

if __name__ == "__main__":
    App().mainloop()
