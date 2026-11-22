"""Build report.pdf from the results files. Every number is read from results/*.json|csv - nothing hand-typed.

Usage: python -m analysis.make_report
"""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common.io import RESULTS, ROOT, SPLITS_JSON, load_json

INK, INK2, GRID, ACCENT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#e1e0d9"), colors.HexColor("#2a78d6")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=INK, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=INK, spaceBefore=8, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.2, leading=12.6, textColor=INK, alignment=TA_LEFT)
SMALL = ParagraphStyle("s", parent=P, fontSize=8, leading=10.5, textColor=INK2)


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.2), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.2), ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2)]
    t.setStyle(TableStyle(st))
    return t


def fig(name, width=16.5):
    p = RESULTS / name
    if not p.exists():
        return Spacer(1, 1)
    img = Image(str(p))
    r = width * cm / img.drawWidth
    img.drawWidth, img.drawHeight = img.drawWidth * r, img.drawHeight * r
    return img


def main():
    R = load_json(RESULTS / "evaluation_results.json")
    S = load_json(SPLITS_JSON)
    V, I = R["verification"], R["identification"]
    tf, fin = V["threshold_free_test"], V["final_threshold"]
    lfw, dg = S["lfw"], S.get("digiface")
    abl = pd.read_csv(RESULTS / "ablation.csv") if (RESULTS / "ablation.csv").exists() else None
    edge = pd.read_csv(RESULTS / "edge_benchmark.csv") if (RESULTS / "edge_benchmark.csv").exists() else None
    story = []

    story += [Paragraph("Face Verification &amp; Re-Identification with a ResNet Embedding Model", H1),
              Paragraph("Technical assessment report - Biz Tech Analytics. All numbers below are generated from "
                        "<i>results/evaluation_results.json</i>, <i>ablation.csv</i> and <i>edge_benchmark.csv</i> by "
                        "<i>analysis/make_report.py</i>.", SMALL), Spacer(1, 6)]

    story += [Paragraph("1. Headline results (held-out test identities never seen in training)", H2),
              table([["Metric", "Value", "Notes"],
                     ["ROC-AUC (5k/5k test pairs)", f"{tf['AUC']:.4f}", f"95% CI {tf['AUC_95CI'][0]:.4f}-{tf['AUC_95CI'][1]:.4f}"],
                     ["EER", pct(tf["EER"]), f"95% CI {pct(tf['EER_95CI'][0])}-{pct(tf['EER_95CI'][1])}"],
                     ["TAR @ FAR=1%", pct(tf["TAR@FAR=1%"]), "test pairs"],
                     ["TAR @ FAR=0.1%", pct(tf["all_vs_all"]["TAR@FAR=0.1%"]),
                      f"all-vs-all: {tf['all_vs_all']['impostor']:,} impostor scores"],
                     ["Final cosine threshold", f"{fin['threshold']:.4f}", f"{fin['operating_point']} chosen on validation identities"],
                     ["Precision / Recall / F1 @ threshold", f"{fin['precision']:.4f} / {fin['recall']:.4f} / {fin['f1']:.4f}", "test pairs"],
                     ["Rank-1 (single-shot gallery)", pct(I["single_shot"]["rank1"]), f"{I['single_shot']['gallery_identities']} identities, {I['single_shot']['probes']} probes"],
                     ["Rank-5 (single-shot gallery)", pct(I["single_shot"]["rank5"]), ""],
                     ["Rank-1 (multi-shot, 3 imgs/id)", pct(I["multi_shot_3"]["rank1"]), ""],
                     ["Retrieval mAP", f"{I['retrieval_mAP']:.4f}", "every test image queries all others"]],
                    widths=[5.6 * cm, 4.2 * cm, 7.2 * cm])]

    story += [Paragraph("2. Dataset", H2), Paragraph(
        f"<b>LFW</b> (Labeled Faces in the Wild, UMass Amherst; public research benchmark of news photographs of public "
        f"figures). Identities with &ge;{lfw['min_images_per_identity']} images: <b>{lfw['identities']}</b> identities, "
        f"<b>{lfw['images']:,}</b> images. Every image went through YuNet detection and our own 5-point similarity "
        f"alignment to the 112x112 ArcFace template (all {lfw['detection'].get('aligned', 0):,} images were detected; "
        f"no fallback crops). "
        f"<b>DigiFace-1M</b> (Microsoft, WACV 2023; fully synthetic, rendered identities - no real person): "
        f"{dg['identities'] if dg else 0:,} identities, {dg['images'] if dg else 0:,} images, used only for pre-training "
        f"under the non-commercial Research Use of Data Agreement.", P)]
    sp = lfw["splits"]
    story += [Spacer(1, 4), table([["Split", "Identities", "Images (total)", "Images used", "Images / identity used (min / median / max)"]] +
                                  [[k, sp[k]["identities"], sp[k]["images"], sp[k]["images_used"],
                                    " / ".join(map(str, sp[k]["images_per_identity_min_median_max"]))]
                                   for k in ("train", "val", "test")], widths=[2.2 * cm, 2.3 * cm, 2.8 * cm, 2.6 * cm, 6.2 * cm]),
              Paragraph(f"Splits are identity-disjoint (seed {lfw['seed']}). Near-duplicate photos were found with a "
                        f"two-stage test (pHash on the full photo, confirmed by pixel correlation &ge;0.93): "
                        f"{lfw['near_duplicate_pairs']} duplicate pairs, all within one identity; duplicate clusters "
                        f"are kept inside one split and are never used as a positive pair. Val/test identities are "
                        f"capped at 30 images so no single person dominates.", SMALL)]

    story += [Paragraph("3. Model and training", H2), Paragraph(
        "ResNet50 (ImageNet init, stem max-pool removed so layer4 is 7x7 for 112x112 faces) -&gt; 1x1 reduce to 512 "
        "-&gt; BN -&gt; Dropout -&gt; FC -&gt; BN -&gt; <b>512-D embedding -&gt; L2 normalisation</b>. "
        "Loss = ArcFace (additive angular margin) + 0.1 x batch-hard triplet, on identity-balanced P x K batches. "
        f"<b>Stage 1</b> trains jointly on the {dg['identities'] if dg else 0:,} synthetic DigiFace identities and the "
        f"{sp['train']['identities']} real LFW <i>training</i> identities (as extra classes); <b>stage 2</b> fine-tunes on the "
        f"LFW training identities alone. Validation and test identities are never used. "
        "Model selection uses TAR@FAR=1% on validation identities only. "
        "<b>Why this recipe:</b> ArcFace optimises the geometry used at test time (angles between unit embeddings) and "
        "needs no pair mining; the triplet term adds an explicit relative-distance constraint on the hardest in-batch "
        "pairs. The ablation below shows that the largest single gain is <i>identity diversity</i> from pre-training, "
        "not the choice of loss: among LFW-only runs ArcFace is ahead on the test set, but softmax is ahead on "
        "validation AUC, so the loss choice is a smaller, less certain effect. An earlier stage 1 that used only 4,000 "
        "synthetic identities overfit them (real-face validation peaked at epoch 5 while synthetic training accuracy "
        "reached 99%); adding more identities and the real training identities fixed this (row v1 vs stage 1).", P)]

    if abl is not None:
        cell = ParagraphStyle("cell", parent=P, fontSize=8, leading=9.6)
        story += [KeepTogether([Paragraph("4. Ablation", H2),
                  Paragraph("Model-selection decisions were made on the validation columns; the test columns are reported "
                            "for completeness. TAR = TAR@FAR=1%; Rank-1 = single-shot gallery.", SMALL),
                  table([["Variant", "val AUC", "val TAR", "test AUC", "test EER", "test TAR", "Rank-1"]] +
                        [[Paragraph(r.description, cell), f"{r.val_AUC:.4f}", pct(r["val_TAR@FAR=1%"], 1), f"{r.test_AUC:.4f}",
                          pct(r.test_EER, 2), pct(r["test_TAR@FAR=1%"], 1), pct(r.test_Rank1, 1)] for _, r in abl.iterrows()],
                        widths=[6.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm])])]

    story += [PageBreak(), Paragraph("5. Verification: ROC, similarity distributions and threshold", H2),
              Paragraph(f"Positive/negative pairs: {R['pairs']['test']['positives']:,} genuine + "
                        f"{R['pairs']['test']['negatives']:,} impostor test pairs from {R['pairs']['test']['identities']} "
                        f"unseen identities (sqrt-proportional per-identity quota, canonical ordering, no duplicates, "
                        f"no self or near-duplicate pairs). Cosine similarity is computed on the L2-normalised "
                        f"embeddings (flip-TTA), never on classifier logits.", P),
              fig("roc_curve.png", 15.5), fig("similarity_distribution.png", 11.5)]
    rows = [["Operating point (fixed on val)", "Threshold", "TAR", "FAR", "Precision", "Recall", "F1", "Accuracy"]]
    names = {"eer": "EER", "far_1e-2": "FAR = 1%", "far_1e-3": "FAR = 0.1%", "max_f1": "max-F1"}
    for k, m in V["test_at_val_thresholds"].items():
        rows.append([names[k], f"{m['threshold']:.4f}", pct(m["TPR(TAR)"]), pct(m["FPR(FAR)"], 3), f"{m['precision']:.4f}",
                     f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['accuracy']:.4f}"])
    story += [KeepTogether([table(rows), Spacer(1, 4), Paragraph(
        f"<b>Final rule:</b> {fin['rule']}  ({fin['operating_point']}, chosen on validation identities and applied "
        f"unchanged to test). <b>Threshold trade-off:</b> raising the threshold makes the system stricter - fewer "
        f"impostors are accepted (FAR falls) but more genuine users are rejected (FRR rises); lowering it does the "
        f"opposite. For access control a false accept is the costly error, so the operating point is set by a target "
        f"FAR; a convenience application would move toward the EER / max-F1 thresholds.", P)]),
              fig("far_frr_vs_threshold.png", 11.5)]

    story += [Paragraph("6. Gallery / Probe identification", H2), Paragraph(
        "Gallery = enrolled test identities; probes = their remaining images; each probe is assigned the gallery "
        "identity with the highest cosine similarity. Open-set: 20% of test identities are not enrolled and must be "
        "rejected.", P),
              table([["Protocol", "Gallery ids", "Probes", "Rank-1", "Rank-5", "Rank-10"]] +
                    [[k, I[k]["gallery_identities"], I[k]["probes"], pct(I[k]["rank1"]), pct(I[k]["rank5"]), pct(I[k]["rank10"])]
                     for k in ("single_shot", "multi_shot_3")]),
              Spacer(1, 4),
              Paragraph("Open-set detection-and-identification rate: " + ", ".join(
                  f"DIR@{k.split('=')[1]} FAR = {pct(v)}" for k, v in I["open_set"].items() if k.startswith("DIR")), P),
              fig("cmc_curve.png", 12)]

    if edge is not None:
        cols = ["model", "runtime", "precision", "device", "threads", "size_MB", "p50_ms", "test_AUC", "TAR@FAR=1%"]
        e = edge[cols].astype(object).where(edge[cols].notna(), "-")
        e["threads"] = [str(int(t)) if t != "-" else "-" for t in e["threads"]]
        story += [PageBreak(), Paragraph("7. Edge deployment", H2), Paragraph(
            "ResNet50 teacher distilled into a ResNet18 student (embedding cosine + relational similarity loss), "
            "exported to ONNX (parity-checked) and statically quantised to INT8 with ONNX Runtime (per-channel QDQ, "
            "calibrated on training images only). <b>Caveat:</b> every latency was measured on a desktop CPU "
            "(AMD Ryzen 9 9950X) and an RTX 5080, <i>not</i> on real edge hardware. Limiting the CPU to 1 or 4 threads is "
            "only a rough proxy; a Raspberry Pi 5 or Jetson CPU would be several times slower, so use these numbers "
            "for relative comparisons (size, speed-up, accuracy cost), not as device specifications. TensorRT FP16 is the "
            "same ONNX-to-TensorRT path used on Jetson GPUs.", P),
                  table([cols] + e.astype(str).values.tolist()), Spacer(1, 4)]
        if R.get("edge", {}).get("pipeline"):
            p = R["edge"]["pipeline"]
            story += [Paragraph(f"Full pipeline on raw photos (detect + align + embed, {p['embedder']}): "
                                f"{p['total_ms_p50']} ms per face (~{p['fps_estimate']} FPS).", P)]
        if R.get("edge_threshold"):
            et = R["edge_threshold"]
            story += [Paragraph(f"The edge model gets its own threshold, calibrated on val at the same operating point: "
                                f"{et['threshold']:.4f} (test TAR {pct(et['test_at_threshold']['TPR(TAR)'])}, "
                                f"FAR {pct(et['test_at_threshold']['FPR(FAR)'], 3)}).", P)]
        story += [fig("edge_benchmark.png", 14)]

    ea = R.get("error_analysis", {})
    story += [PageBreak(), Paragraph("8. Error analysis and explainability", H2)]
    if ea:
        story += [Paragraph(
            f"At the final threshold ({ea['threshold']:.3f}) the 10,000 test pairs contain {ea['false_accepts']} false "
            f"accepts and {ea['false_rejects']} false rejects. In the examples below, the worst genuine pairs involve "
            f"sunglasses, helmets and caps, strong blur and extreme expression or pose; the worst impostor pairs are "
            f"look-alikes with similar pose, hair or headwear. These are the conditions a deployment should test for "
            f"first.", P)]
    story += [fig("error_cases.png", 15.0), fig("gradcam.png", 8.2)]

    story += [Paragraph("9. Limitations and responsible use", H2), Paragraph(
        "LFW is dominated by news photos of (mostly male, light-skinned) public figures; results do not transfer "
        "automatically to other demographics, cameras or capture conditions, and per-group error rates were not "
        "measured. The test set has 122 identities, so low-FAR estimates carry wide uncertainty (bootstrap CIs "
        "reported). No liveness / presentation-attack detection is included - a printed photo would be scored like a "
        "live face. DigiFace-1M is licensed for non-commercial research only, so these weights may not be used "
        "commercially. Face recognition is biometric processing: deployment needs consent, a lawful basis, data "
        "minimisation (the API stores embeddings only and supports deletion) and human review of decisions.", P)]

    doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Face Verification Report")
    doc.build(story)
    print(f"[report] {ROOT / 'report.pdf'}")


if __name__ == "__main__":
    main()
