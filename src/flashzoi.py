import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.amp import autocast
from pathlib import Path

from borzoi_pytorch import Borzoi

# ─── CONFIG ──────────────────────────────────────────────────────────
hap_ids      = ["NA20826_chr22_hap2", "HG00188_chr22_hap1", "HG00369_chr22_hap1"]
fastas_dir   = Path("/blue/juannanzhou/fahimeh.rahimi/flashzoi/chr22_fastas_snp")
alts_root    = Path("/orange/juannanzhou/UTR_alts")
variants_csv = Path("/blue/juannanzhou/fahimeh.rahimi/flashzoi/Filtered_chr22_SNPs.csv")
model_name   = "johahi/flashzoi-replicate-0"
outdir       = Path("outputs")
outdir.mkdir(exist_ok=True)

# ─── PREP ────────────────────────────────────────────────────────────
vdf = pd.read_csv(variants_csv, dtype={'CHROM':str})
vdf = vdf[vdf.CHROM.str.lstrip("chr")=="22"]
lookup = {(str(r.POS), r.REF, r.ALT): r.CLNSIG for r in vdf.itertuples()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Borzoi.from_pretrained(model_name).to(device).eval()

# ─── HELPERS ─────────────────────────────────────────────────────────
def one_hot_seq(seq: str) -> np.ndarray:
    mapping = {'A':0,'C':1,'G':2,'T':3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = mapping.get(b)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

def score_all_tracks(seq: str) -> np.ndarray:
    x = torch.from_numpy(one_hot_seq(seq)[None]).to(device).half()
    with torch.no_grad(), autocast("cuda", torch.float16):
        # model(x) shape: (1, n_tissues, L); we take [0]
        return model(x)[0].cpu().numpy()

# ─── MAIN ────────────────────────────────────────────────────────────
records = []
half_win = None
pat = re.compile(r'.*_22_(\d+)_([ACGT])_([ACGT])_alt$')

for hap_id in tqdm(hap_ids, desc="Haplotypes"):
    fasta_path = fastas_dir / f"{hap_id}.fa"
    alts_dir   = alts_root / hap_id
    full_seq   = "".join(fasta_path.read_text().splitlines()[1:]).upper()
    L = len(full_seq)

    # determine half window once
    if half_win is None:
        first_alt = next(alts_dir.glob("*.fa"))
        seq1 = first_alt.read_text().splitlines()[1].upper()
        half_win = (len(seq1)-1)//2

    for alt_fa in tqdm(sorted(alts_dir.glob("*.fa")), desc=f"{hap_id} variants", leave=False):
        stem = alt_fa.stem
        m = pat.match(stem)
        if not m:
            continue
        pos_s, ref, alt = m.groups()
        pos = int(pos_s) - 1

        seq = alt_fa.read_text().splitlines()[1].upper()
        start, end = pos-half_win, pos+half_win+1
        if start < 0 or end > L:
            continue

        # score reference and alternate
        ref_seq = full_seq[start:end]
        alt_seq = seq
        ref_t = score_all_tracks(ref_seq)
        alt_t = score_all_tracks(alt_seq)
        delta = alt_t - ref_t

        mean_d = float(delta.mean())
        cln = lookup.get((pos_s, ref, alt), np.nan)

        records.append({
            'hap_id':     hap_id,
            'variant':    f"{pos_s}_{ref}>{alt}",
            'CLNSIG':     cln,
            'mean_delta': mean_d
        })

# build DataFrame and save
df = pd.DataFrame(records)
summary_csv = outdir / "summary_3haps_all_tissues.csv"
df.to_csv(summary_csv, index=False)
print(f"Saved summary to {summary_csv}")

