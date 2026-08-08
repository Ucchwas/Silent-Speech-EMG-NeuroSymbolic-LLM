"""
Tune the NeuroSymbolic decoding hyperparameters on the VALIDATION split.

Two stages, because only some parameters change the search itself:

  Stage A (GPU)  min_eos_len / beam_size gate which hypotheses can ever be
                 formed, so each value needs its own run of Algorithm 2.
                 For every setting we keep the whole completed-candidate pool.

  Stage B (CPU)  beta / kappa / gamma / alpha / delta only re-rank an existing
                 pool, so the grid is swept offline over the cached pools.

Nothing here ever reads the test split.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from inference_emg_llm import (
    load_model_and_tt,
    read_text_from_json,
    scan_pairs,
    tidy,
)
from scripts.ar_decode import PrefixCachedAR, ctc_greedy_decode_ids
from scripts.dataset_emg import load_normalizer
from scripts.metrics import score_corpus
from scripts.ns_utils import (
    NSConfig,
    build_trie_from_lexicon,
    build_grammar_trie,
    compute_adaptive_lambda,
    ctc_forward_logprob,
    generate_candidates,
    select_with_ctc,
    train_char_5gram,
)


def parse_args():
    p = argparse.ArgumentParser()
    # Required, not defaulted: the tuned config is only valid for the exact
    # checkpoint it was searched on (the beam optimum differs per backbone --
    # K=48 for Llama-3.2-1B, K=12 for Llama-3.2-3B), so silently falling back
    # to some other model's checkpoint would produce a config that looks fine
    # and is wrong.
    p.add_argument("--ckpt", required=True, help="Checkpoint to tune decoding for.")
    p.add_argument("--data_dir", default="data/val_emg")
    p.add_argument("--normalizer", default="artifacts/emg_norm.pkl")
    p.add_argument("--lexicon", default="artifacts/lexicon.txt")
    p.add_argument("--train_dir_for_lm", default="data/train_emg")
    p.add_argument("--ngram_alpha", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--beam", type=int, default=6)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--per_step_top", type=int, default=50)
    p.add_argument("--rerank_M", type=int, default=16)
    p.add_argument("--min_eos_grid", default="2,4,8")
    # Beam width turned out to be the dominant decoding parameter. Under the
    # utterance grammar the search is constrained hard enough that K=6 leaves a
    # pool of only ~2.5 completions and the reference is reachable just 40% of
    # the time; at K=48 the pool is ~23 and reachability rises to 70%, taking
    # AR+CTC reranking from 0.308 to 0.233 on validation.
    p.add_argument("--beam_grid", default="6,12,24,48")
    p.add_argument("--grammar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default="artifacts/ns_tuning.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, tt, _ = load_model_and_tt(Path(args.ckpt), device)
    norm = load_normalizer(args.normalizer)
    trie_root = (build_grammar_trie(args.train_dir_for_lm, tt) if args.grammar
                 else build_trie_from_lexicon(tt, lexicon_path=args.lexicon))
    charlm = train_char_5gram(tt, train_dir=args.train_dir_for_lm, alpha=args.ngram_alpha)

    pairs = scan_pairs(Path(args.data_dir))
    print(f"val utterances: {len(pairs)}", flush=True)

    grid_m = [int(v) for v in args.min_eos_grid.split(",")]
    grid_k = [int(v) for v in args.beam_grid.split(",")]
    keys = [(k, m) for k in grid_k for m in grid_m]

    # ---- Stage A: one search per min_eos_len, caching the candidate pools ----
    # pools[m][i] is the completed-hypothesis pool for utterance i.
    pools: dict[tuple, list] = {key: [] for key in keys}
    refs: list[str] = []
    ctc_texts: list[str] = []
    ctc_logits_all: list[torch.Tensor] = []
    ar_beams: list[str] = []
    scorers: list = []

    for k, (npy_path, json_path) in enumerate(pairs):
        x = np.load(npy_path).astype(np.float32)
        x = norm.transform(x).astype(np.float32)
        emg = torch.from_numpy(x).unsqueeze(0).to(device)
        emg_mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
        refs.append(tt.clean(read_text_from_json(json_path)))

        prefix_embeds, prefix_mask = model.prepare_context(emg, emg_mask)
        ctc_logits = model.ctc_posteriors(emg, emg_mask)
        ctc_logits_all.append(ctc_logits.detach().cpu())
        ctc_texts.append(
            tidy(tt.ids_to_text(ctc_greedy_decode_ids(ctc_logits, blank_id=tt.PAD_IDX)))
        )

        dec = PrefixCachedAR(model, tt, prefix_embeds, prefix_mask, capacity=max(max(grid_k), 2))
        scorers.append(dec)
        ar_beams.append(tidy(tt.ids_to_text(dec.beam(
            beam_size=args.beam, max_len=args.max_len, alpha=0.6, delta=5.0))))

        for (K, m) in keys:
            cfg = NSConfig(
                beam_size=K, max_len=args.max_len, top_m=max(64, 4 * K),
                min_eos_len=m, per_step_top=max(args.per_step_top, 4 * K),
            )
            pools[(K, m)].append(generate_candidates(dec.step, tt, trie_root, charlm, cfg))
        if (k + 1) % 10 == 0:
            print(f"  searched {k+1}/{len(pairs)}", flush=True)

    base = score_corpus(refs, ar_beams)
    print(f"\nval AR beam baseline: WER {base.wer:.4f} CER {base.cer:.4f}", flush=True)

    # ---- Stage B: offline rescoring grid ------------------------------------
    results = []
    grid = itertools.product(
        keys,
        [0.0, 0.3, 0.55, 0.8],      # beta
        [0.0, 0.2, 0.4],            # kappa
        [0.45, 1.0, 2.0],           # gamma
        [0.0, 0.6],                 # alpha
    )
    for (K, m), beta, kappa, gamma, alpha in grid:
        cfg = NSConfig(
            beam_size=K, max_len=args.max_len, top_m=max(64, 4 * K),
            min_eos_len=m, per_step_top=max(args.per_step_top, 4 * K),
            beta=beta, kappa=kappa, gamma=gamma, alpha=alpha, delta=5.0,
        )
        hyps = []
        for pool in pools[(K, m)]:
            if not pool:
                hyps.append("")
                continue
            best = max(pool, key=lambda c: c.score(cfg, 0.0, 0.0))
            hyps.append(tidy(best.text))
        s = score_corpus(refs, hyps)
        results.append(
            dict(beam=K, min_eos_len=m, beta=beta, kappa=kappa, gamma=gamma, alpha=alpha,
                 wer=s.wer, cer=s.cer, em=s.exact_match)
        )

    results.sort(key=lambda r: r["wer"])
    print("\ntop 15 NS (constrained beam) configs on validation:")
    for r in results[:15]:
        print(f"  K={r['beam']} m={r['min_eos_len']} beta={r['beta']:.2f} kappa={r['kappa']:.2f} "
              f"gamma={r['gamma']:.2f} alpha={r['alpha']:.1f}  "
              f"WER {r['wer']:.4f} CER {r['cer']:.4f} EM {r['em']:.0f}%")

    # ---- Stage B2: choose the beam width on the FINAL objective -------------
    # Selecting K by NS-alone WER would be wrong here. Measured on validation,
    # NS-alone is flat-to-worse as the beam widens (0.3233 at K=6 -> 0.3308 at
    # K=48) while AR+CTC reranking over the same pools improves sharply
    # (0.3083 -> 0.2331), because a wider beam raises how often the reference is
    # reachable at all (40% -> 70%). K is therefore picked on the joint score,
    # with the per-(K,m) symbolic weights that are best for that pool.
    best_by_key = {}
    for r in results:
        k = (r["beam"], r["min_eos_len"])
        if k not in best_by_key or r["wer"] < best_by_key[k]["wer"]:
            best_by_key[k] = r

    def _ctc_rows(key, ns_cfg):
        rows_all, ns_txt = [], []
        for i, pool in enumerate(pools[key]):
            if not pool:
                rows_all.append([]); ns_txt.append("")
                continue
            ns_txt.append(tidy(max(pool, key=lambda c: c.score(ns_cfg, 0.0, 0.0)).text))
            lg = ctc_logits_all[i].to(device)
            blank_id = int(getattr(tt, "PAD_IDX", lg.shape[-1] - 1))
            rows_all.append([
                (c, float(ctc_forward_logprob(lg, tt.text_to_ids(c.text.strip(), add_bos_eos=False),
                                              blank_id=blank_id)))
                for c in pool
            ])
        return rows_all, ns_txt

    lam_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    sel = None
    print("\nbeam selection on the joint AR+CTC objective:")
    for key, r in sorted(best_by_key.items()):
        K, m = key
        cfg_k = NSConfig(
            beam_size=K, max_len=args.max_len, top_m=max(64, 4 * K),
            min_eos_len=m, per_step_top=max(args.per_step_top, 4 * K),
            beta=r["beta"], kappa=r["kappa"], gamma=r["gamma"], alpha=r["alpha"], delta=5.0,
        )
        rows_all, _ = _ctc_rows(key, cfg_k)
        for lam in lam_grid:
            hyps = [tidy(max(rw, key=lambda t: t[0].score(cfg_k, t[1], lam))[0].text) if rw else ""
                    for rw in rows_all]
            w = score_corpus(refs, hyps).wer
            if sel is None or w < sel[0]:
                sel = (w, key, r, lam)
        print(f"  K={K:3d} m={m}: NS-alone {r['wer']:.4f}  best joint {sel[0]:.4f} (running best)")

    _, (k_best, m_best), best, lam_fix_pre = sel
    print(f"\nchosen: K={k_best} m={m_best} lambda~{lam_fix_pre} -> joint WER {sel[0]:.4f}")

    # ---- Stage C: CTC-assisted stages at the chosen operating point ---------
    # ctc_forward_logprob does not depend on lambda, so it is computed once per
    # candidate and every lambda below is then a free re-ranking.
    cfg = NSConfig(
        beam_size=k_best, max_len=args.max_len, top_m=max(64, 4 * k_best),
        min_eos_len=m_best, per_step_top=max(args.per_step_top, 4 * k_best),
        beta=best["beta"], kappa=best["kappa"], gamma=best["gamma"],
        alpha=best["alpha"], delta=5.0,
    )

    ns_texts = []
    cand_cache = []   # per utterance: list of (candidate, ctc_logp)
    for i, pool in enumerate(pools[(k_best, m_best)]):
        if not pool:
            ns_texts.append("")
            cand_cache.append([])
            continue
        ns_texts.append(tidy(max(pool, key=lambda c: c.score(cfg, 0.0, 0.0)).text))
        lg = ctc_logits_all[i].to(device)
        blank_id = int(getattr(tt, "PAD_IDX", lg.shape[-1] - 1))
        rows = []
        for c in pool:
            labels = tt.text_to_ids(c.text.strip(), add_bos_eos=False)
            rows.append((c, float(ctc_forward_logprob(lg, labels, blank_id=blank_id))))
        cand_cache.append(rows)

    # --- ns_ctc_select: Eq. 18, with and without length normalisation --------
    sel_best = None
    for ln in (True, False):
        hyps = [
            tidy(select_with_ctc(ns_texts[i], ctc_texts[i], tt, scorers[i].score,
                                 length_norm=ln, cfg=cfg)[0])
            for i in range(len(pairs))
        ]
        s = score_corpus(refs, hyps)
        print(f"  ns_ctc_select length_norm={ln}: WER {s.wer:.4f}")
        if sel_best is None or s.wer < sel_best[0]:
            sel_best = (s.wer, ln, hyps)
    sel_texts = sel_best[2]

    # --- ns_joint_rerank: sweep the fixed lambda -----------------------------
    lam_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    rr_best = None
    for lam in lam_grid:
        hyps = []
        for rows in cand_cache:
            if not rows:
                hyps.append(""); continue
            c, _ = max(rows, key=lambda r: r[0].score(cfg, r[1], lam))
            hyps.append(tidy(c.text))
        s = score_corpus(refs, hyps)
        print(f"  ns_joint_rerank lambda={lam:.2f}: WER {s.wer:.4f}")
        if rr_best is None or s.wer < rr_best[0]:
            rr_best = (s.wer, lam, hyps)
    rr_texts = rr_best[2]
    lam_fix = rr_best[1]

    # --- ns_joint_adaptive: sweep (lambda0, a, b) around the fixed optimum ----
    # The clip range has to be wide enough to contain the fixed optimum, or the
    # adaptive branch cannot even reach the point ns_joint_rerank already found.
    # It is computed once here and written into the tuned config below: leaving
    # it implicit is what made inference fall back to lambda_max = 0.6 and clip
    # a tuned lambda0 of 1.9, so ns_joint_adaptive scored *worse* than
    # ns_joint_rerank rather than at least matching it.
    lam_lo, lam_hi = 0.0, max(0.6, lam_fix + 0.3)
    ad_best = None
    for lam0 in sorted({max(0.0, lam_fix - 0.1), lam_fix, lam_fix + 0.1}):
        for a in (0.0, 0.05, 0.1, 0.2):
            for b in (0.0, 0.05, 0.1, 0.2):
                cfg_a = replace(cfg, lambda0=lam0, a=a, b=b,
                                lambda_min=lam_lo, lambda_max=lam_hi)
                hyps = []
                for i, rows in enumerate(cand_cache):
                    if not rows:
                        hyps.append(""); continue
                    ref_c = max(rows, key=lambda r: r[0].score0(cfg))[0]
                    lam, _, _ = compute_adaptive_lambda(
                        ref_c, ctc_logits_all[i].to(device), tt, cfg_a
                    )
                    c, _ = max(rows, key=lambda r: r[0].score(cfg, r[1], lam))
                    hyps.append(tidy(c.text))
                s = score_corpus(refs, hyps)
                if ad_best is None or s.wer < ad_best[0]:
                    ad_best = (s.wer, dict(lambda0=lam0, a=a, b=b), hyps)
    print(f"  ns_joint_adaptive best: WER {ad_best[0]:.4f} at {ad_best[1]}")
    ad_texts = ad_best[2]

    print("\n=== validation ladder at tuned config ===")
    ladder = [("ctc_greedy", ctc_texts), ("ar_beam", ar_beams), ("ns", ns_texts),
              ("ns_ctc_select", sel_texts), ("ns_joint_rerank", rr_texts),
              ("ns_joint_adaptive", ad_texts)]
    for name, hyps in ladder:
        s = score_corpus(refs, hyps)
        print(f"  {name:20s} WER {s.wer:.4f} CER {s.cer:.4f} EM {s.exact_match:.0f}%")

    tuned = dict(
        beam=k_best, min_eos_len=m_best, beta=best["beta"], kappa=best["kappa"],
        gamma=best["gamma"], alpha=best["alpha"], delta=5.0,
        ctc_select_lengthnorm=sel_best[1], lambda_fix=lam_fix,
        lambda_min=lam_lo, lambda_max=lam_hi, **ad_best[1],
    )
    print(f"\ntuned NS config: {json.dumps(tuned)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(tuned=tuned, ns_grid=results[:60]), open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
