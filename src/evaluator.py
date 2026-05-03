import torch
from torch import device, nn
from .parsers import wav_to_logmel, parse_phonemes
from .constants import Constants as C


def evaluate_audio(
    wav_path,
    textgrid_path=None,
    model=None,
    device=None,
    collapse=True,
    show_per_window=False,
    top_k=3,
):
    """Predict the phoneme sequence in one audio file, with probabilities.

    top_k: how many candidate phonemes to show per window (set 0 to hide).
    """
    if model is None:
        raise ValueError("model must be provided")
    if device is None:
        # Default to the model's device to avoid cross-device errors.
        device = next(model.parameters()).device
    else:
        model = model.to(device)
    # 1. compute mel spectrogram
    mel = wav_to_logmel(wav_path)
    n_mels, T = mel.shape
    frame_dur = C.HOP_LENGTH / C.SAMPLE_RATE

    # 2. slide windows
    windows, times = [], []
    for start in range(0, T - C.WIN_FRAMES + 1, C.SHIFT_FRAMES):
        end = start + C.WIN_FRAMES
        windows.append(mel[:, start:end])
        times.append((start * frame_dur, end * frame_dur))
    if not windows:
        print("audio too short for one window")
        return None
    X = torch.stack(windows).to(device)

    # 3. predict + probabilities
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu()  # (W, N_CLASSES)
    pred_idx = probs.argmax(1).tolist()
    pred = [C.IDX2LABEL[i] for i in pred_idx]
    pred_prob = [probs[i, pred_idx[i]].item() for i in range(len(pred_idx))]

    # 4. optional ground truth
    truth = None
    if textgrid_path is not None:
        with open(textgrid_path, "r", encoding="utf-8") as f:
            phonemes = parse_phonemes(f.read())
        truth = []
        for t_start, t_end in times:
            best, best_overlap = None, 0.0
            for pmin, pmax, ptext in phonemes:
                if ptext == "" or ptext not in C.LABEL2IDX:
                    continue
                overlap = min(pmax, t_end) - max(pmin, t_start)
                if overlap > best_overlap:
                    best_overlap, best = overlap, ptext
            truth.append(best if best is not None else "?")

    # 5. print
    print(f"\nfile: {wav_path}")
    print(f"duration: {T * frame_dur:.2f}s, {len(pred)} windows")

    if show_per_window:
        print("\nper-window predictions:")
        header = f"{'start':>6} {'end':>6}  {'pred':>6} {'p':>5}"
        if top_k > 1:
            header += f"   top-{top_k}"
        if truth is not None:
            header += "   truth"
        print(header)

        for i, (t0, t1) in enumerate(times):
            row = f"{t0:>6.2f} {t1:>6.2f}  {pred[i]:>6} {pred_prob[i]:>5.2f}"

            if top_k > 1:
                # top-k candidates: 'a:0.62 e:0.18 o:0.09'
                topv, topi = probs[i].topk(top_k)
                cand = " ".join(
                    f"{C.IDX2LABEL[j.item()]}:{v.item():.2f}"
                    for v, j in zip(topv, topi)
                )
                row += f"   {cand}"

            if truth is not None:
                # also show probability the model gave to the *true* class
                t_idx = C.LABEL2IDX[truth[i]] if truth[i] in C.LABEL2IDX else None
                t_p = probs[i, t_idx].item() if t_idx is not None else float("nan")
                mark = "✓" if pred[i] == truth[i] else "✗"
                row += f"   {truth[i]}({t_p:.2f}) {mark}"

            print(row)

    # 6. collapse runs, with avg probability per run
    def runs_with_prob(seq, ps):
        out = []
        for s, p in zip(seq, ps):
            if not out or out[-1][0] != s:
                out.append([s, [p]])
            else:
                out[-1][1].append(p)
        # convert list of probs to (count, mean_prob)
        return [(s, len(ps_), sum(ps_) / len(ps_)) for s, ps_ in out]

    if collapse:
        pr = runs_with_prob(pred, pred_prob)
        print("\npredicted phoneme sequence (count, avg prob):")
        print("  " + " ".join(f"{p}({n},{avg:.2f})" for p, n, avg in pr))

    # 7. accuracy + average confidence
    avg_conf = sum(pred_prob) / len(pred_prob)
    print(f"\naverage confidence on predicted class: {avg_conf:.3f}")

    if truth is not None:
        correct = sum(p == t for p, t in zip(pred, truth))
        acc = correct / len(pred)
        print(f"window-level accuracy: {correct}/{len(pred)} = {acc:.3f}")
        return {
            "pred": pred,
            "pred_prob": pred_prob,
            "probs": probs.numpy(),
            "truth": truth,
            "times": times,
            "accuracy": acc,
        }

    return {
        "pred": pred,
        "pred_prob": pred_prob,
        "probs": probs.numpy(),
        "times": times,
    }
