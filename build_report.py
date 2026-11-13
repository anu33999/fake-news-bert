"""Builds the submission report PDF from the executed notebook outputs."""
import json, re, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, Preformatted, KeepTogether)

O = "outputs"
M = json.load(open(f"{O}/metrics.json")); H = json.load(open(f"{O}/history.json"))
R = pd.read_csv(f"{O}/results.csv", index_col=0)
ACC = pd.read_csv(f"{O}/acc_by_label.csv", index_col=0).iloc[:, 0]
SAMP = pd.read_csv(f"{O}/sample_predictions.csv")
b, lr, mj = R.loc["BERT (fine-tuned)"], R.loc["TF-IDF + LogReg"], R.loc["Majority class"]
cm_ = M["confusion_matrix"]

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1f3b63"), spaceBefore=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1f3b63"))
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=10, leading=14, alignment=4)
BL = ParagraphStyle("bl", parent=P, leftIndent=14, bulletIndent=4)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8)
CAP = ParagraphStyle("cap", parent=P, fontSize=8.5, textColor=colors.grey, alignment=1)

CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11)
def tbl(data, widths=None, head=True):
    if widths: data = [data[0]] + [[Paragraph(c, CELL) if isinstance(c, str) and len(c) > 45 else c for c in row] for row in data[1:]] if head else [[Paragraph(c, CELL) if isinstance(c, str) and len(c) > 45 else c for c in row] for row in data]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa9bb")), ("FONTSIZE", (0, 0), (-1, -1), 9),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b63")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

def img(path, w=16.5, cap=None):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); el = [Image(path, w * cm, w * cm * ih / iw)]
    if cap: el.append(Paragraph(cap, CAP))
    return KeepTogether(el + [Spacer(1, 6)])

bullets = lambda items: [Paragraph(x, BL, bulletText="•") for x in items]
s = []

# ---- Title page
s += [Spacer(1, 3 * cm), Paragraph("Fake News Classification using a BERT Transformer", ParagraphStyle("t", parent=H1, fontSize=22, alignment=1, leading=28)),
      Spacer(1, 6), Paragraph("Deep Learning Case Study 13 — LIAR Dataset", ParagraphStyle("st", parent=P, fontSize=13, alignment=1)), Spacer(1, 1.5 * cm),
      tbl([["Project title", "Fake News Classification using BERT (LIAR dataset)"], ["Student name", "Anwesha Mahesh Mulge"],
           ["Registration number", "23FE10ITE00008"], ["Roll number", "65 (Group 5)"],
           ["Assigned case study", "Case Study 13 — Fake News Classification"], ["Assigned architecture", "BERT Transformer (bert-base-uncased)"],
           ["Framework", "Python, PyTorch + Hugging Face Transformers, scikit-learn"]], [5 * cm, 10 * cm], head=False),
      Spacer(1, 1.2 * cm),
      Paragraph(f"<b>Headline result:</b> the fine-tuned BERT model reaches <b>{b.accuracy:.1%} accuracy</b>, macro-F1 <b>{b.f1_macro:.3f}</b> and ROC-AUC "
                f"<b>{b.roc_auc:.3f}</b> on the official LIAR test set (binary Fake vs Real). It is far ahead of the majority-class baseline "
                f"({mj.accuracy:.1%} accuracy, macro-F1 {mj.f1_macro:.3f}) and slightly ahead of TF-IDF + Logistic Regression on accuracy and macro-F1 "
                f"({lr.accuracy:.1%}, {lr.f1_macro:.3f}). The two are roughly level on ROC-AUC ({lr.roc_auc:.3f} for TF-IDF), which is discussed in Section 6.", P),
      PageBreak()]

# ---- 1 Problem
s += [Paragraph("1. Problem Statement", H1),
      Paragraph("The task is to classify short political statements by truthfulness, using the labels in the LIAR dataset. "
                "The six fine-grained PolitiFact ratings are hard to tell apart even for people, so this project maps them to a binary "
                "<b>Fake vs Real</b> task (Section 2.3). It then fine-tunes a pretrained BERT encoder and compares it with classical baselines.", P)]

# ---- 2 Dataset
tr = pd.read_csv("data/train.tsv", sep="\t", header=None, quoting=3)
dist = tr[1].value_counts()
order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
s += [Paragraph("2. Dataset", H1), Paragraph("2.1 Source", H2),
      Paragraph("LIAR (W. Y. Wang, <i>“Liar, Liar Pants on Fire”</i>, ACL 2017). Download: "
                "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip. It has 12,836 statements collected from PolitiFact.com (2007–2016), "
                "with the official splits train = 10,240, validation = 1,284, test = 1,267.", P),
      Paragraph("2.2 Description", H2),
      Paragraph("Each row has 14 tab-separated fields: statement ID, label, statement text, subject(s), speaker, speaker job, state, party, "
                "five columns of the speaker's earlier rating counts, and the context or venue (for example, “a TV ad”). Statements are short, "
                "about 18 words on average.", P),
      tbl([["Original label"] + order, ["Train count"] + [str(dist.get(k, 0)) for k in order],
           ["Binary label"] + ["Fake (0)"] * 3 + ["Real (1)"] * 3], None),
      Spacer(1, 6), Paragraph("2.3 Label mapping and class imbalance", H2),
      Paragraph("<b>Fake (0)</b> = pants-fire, false, barely-true; <b>Real (1)</b> = half-true, mostly-true, true. "
                "This split puts “half-true” on the Real side, as is common in LIAR binary studies. It gives a mild imbalance: "
                f"about {tr[1].map({k: int(i >= 3) for i, k in enumerate(order)}).mean():.1%} of training statements are Real. "
                "The imbalance is handled with class-weighted cross-entropy and is reported through macro-averaged metrics.", P),
      img(f"{O}/class_distribution.png", 15, "Figure 1 — Class distribution in the training split (6-way and binary)."),
      Paragraph("2.4 Preprocessing", H2)]
s += bullets(["Loaded the TSV files with fixed column names and no quote parsing, because statements contain raw quote characters.",
              "Mapped the labels to binary as described above, using the official splits unchanged with no shuffling across splits.",
              "Built the input text as: <i>statement [SEP] subject: … ; speaker: … ; party: … ; context: …</i>. Missing values became “unknown”.",
              "Tokenized with the <i>bert-base-uncased</i> WordPiece tokenizer (lower-casing, sub-word splitting, [CLS]/[SEP] tokens), "
              "truncated and padded to 96 tokens. That covers more than 95% of inputs, and attention masks mark the padding.",
              "Baseline: TF-IDF on word uni- and bi-grams (20k features, sublinear TF, min_df = 2) over the same text."])

# ---- 3 Architecture
s += [PageBreak(), Paragraph("3. Architecture and Model Summary", H1),
      img(f"{O}/architecture.png", 17, "Figure 2 — Architecture of the fine-tuned BERT classifier."),
      tbl([["Component", "Configuration"], ["Pretrained encoder", "bert-base-uncased (12 layers, 768 hidden, 12 attention heads, 3072 FFN)"],
           ["Embeddings", "WordPiece token (30,522 vocab) + learned position (512) + segment, LayerNorm, dropout 0.1"],
           ["Pooling", "[CLS] token → pooler (Dense 768→768, tanh)"], ["Classification head", "Dropout 0.1 → Dense 768→2 → softmax"],
           ["Total / trainable parameters", f"{M['params_total']:,} / {M['params_total']:,} (full fine-tuning)"],
           ["Loss", "Class-weighted cross-entropy"], ["Optimizer", "AdamW, lr 2e-5, weight decay 0.01, gradient clip 1.0"],
           ["Schedule", "Linear warm-up for 10% of steps, then linear decay"], ["Batch / max length / epochs", "16 / 96 tokens / 3"],
           ["Model selection", "Checkpoint with the best validation macro-F1"], ["Hardware", f"Apple M2 ({M['device'].upper()}), training time ≈ {M['train_minutes']} min"]],
          [5 * cm, 11.5 * cm]),
      Spacer(1, 8),
      Paragraph("Self-attention lets every token attend to every other token. BERT can therefore model negation, hedging words, numbers and "
                "the link between a claim and its speaker or context, which bag-of-words models cannot represent.", P)]

# ---- 4 Training
s += [Paragraph("4. Training and Validation Curves", H1),
      img(f"{O}/training_curves.png", 17, "Figure 3 — Training/validation loss (left) and accuracy and macro-F1 (right) per epoch."),
      KeepTogether(tbl([["Epoch", "Train loss", "Train acc", "Val loss", "Val acc", "Val macro-F1"]] +
          [[str(i + 1), f"{H['train_loss'][i]:.4f}", f"{H['train_acc'][i]:.4f}", f"{H['val_loss'][i]:.4f}", f"{H['val_acc'][i]:.4f}", f"{H['val_f1'][i]:.4f}"]
           for i in range(len(H["train_loss"]))])),
      Spacer(1, 6),
      Paragraph("Training loss falls steadily. Validation loss bottoms out early and then rises while training accuracy keeps climbing, "
                "which is the usual sign of overfitting when a 110M-parameter model sees only 10k short examples. Selecting the checkpoint by "
                "validation macro-F1 guards against this.", P)]

# ---- 5 Test results
s += [PageBreak(), Paragraph("5. Final Test Results", H1),
      tbl([["Metric", "Value"], ["Accuracy", f"{b.accuracy:.4f}"], ["Macro precision", f"{b.precision_macro:.4f}"],
           ["Macro recall", f"{b.recall_macro:.4f}"], ["Macro F1", f"{b.f1_macro:.4f}"], ["ROC-AUC", f"{b.roc_auc:.4f}"]], [6 * cm, 4 * cm]),
      Spacer(1, 8),
      img(f"{O}/confusion_matrix.png", 8.5, f"Figure 4 — Test confusion matrix (rows = actual). TN={cm_[0][0]}, FP={cm_[0][1]}, FN={cm_[1][0]}, TP={cm_[1][1]}."),
      Paragraph("Domain-specific evaluation: accuracy by original LIAR label", H2),
      tbl([["Original label"] + order, ["BERT accuracy"] + [f"{ACC[k]:.3f}" for k in order]]),
      Spacer(1, 6),
      Paragraph("The model is most reliable at the ends of the scale. Accuracy is 79% on true and mostly-true and 75% on pants-fire. "
                "It is weakest on barely-true (50%) and false (56%). Barely-true sits right at the Fake/Real cut, so it is close in meaning to "
                "half-true. Many “false” statements are fluent, specific-sounding claims that resemble true ones on the surface. "
                "Overall the model leans towards predicting Real: 240 Fake statements were labelled Real, against 198 Real statements labelled Fake.", P),
      Paragraph("Sample test predictions", H2),
      tbl([["Statement", "True label", "Pred.", "P(Real)"]] +
          [[Paragraph(r.statement[:150] + ("…" if len(r.statement) > 150 else ""), ParagraphStyle("c", fontSize=8, leading=10)),
            r.label, ["Fake", "Real"][r.pred], f"{r.p_real:.2f}"] for r in SAMP.itertuples()], [10.5 * cm, 2.2 * cm, 1.4 * cm, 1.5 * cm])]

# ---- 6 Baselines
s += [PageBreak(), Paragraph("6. Baseline Comparison", H1),
      tbl([["Model", "Accuracy", "Macro-P", "Macro-R", "Macro-F1", "ROC-AUC"]] +
          [[n, f"{r.accuracy:.4f}", f"{r.precision_macro:.4f}", f"{r.recall_macro:.4f}", f"{r.f1_macro:.4f}",
            "—" if pd.isna(r.roc_auc) else f"{r.roc_auc:.4f}"] for n, r in R.iterrows()]),
      Spacer(1, 8), img(f"{O}/baseline_comparison.png", 17, "Figure 5 — Test metrics by model (left) and ROC curves (right)."),
      Paragraph(f"Compared with TF-IDF + Logistic Regression, BERT changes accuracy by {b.accuracy - lr.accuracy:+.3f}, macro-F1 by {b.f1_macro - lr.f1_macro:+.3f} "
                f"and ROC-AUC by {b.roc_auc - lr.roc_auc:+.3f}. On this test set of 1,283 statements, gaps that small are within random variation, "
                "so the honest conclusion is that the two models perform about equally. "
                f"The majority-class model scores {mj.accuracy:.1%} accuracy but only {mj.f1_macro:.3f} macro-F1, because it never predicts Fake. "
                "Accuracy alone is therefore misleading on this dataset. "
                "The result fits published LIAR studies, where deep models gain little over strong linear baselines. Much of the usable signal is in "
                "speaker, party and context words, which TF-IDF already captures, rather than in deeper reading of the claim. "
                "BERT's benefit would likely grow with evidence text, more training data or several averaged runs.", P)]

# ---- 7 Limitations
s += [Paragraph("7. Discussion of Limitations", H1)]
s += bullets(["<b>Ceiling set by the data:</b> statements are short and given without evidence, so whether a claim is true often cannot be "
              "decided from the text alone. Both models stay well below 80% accuracy, which matches the literature.",
              "<b>Binary mapping:</b> merging six ordinal labels hides real differences, and the cut between barely-true and half-true is a choice. "
              "Other mappings, or 6-way and ordinal regression, would give different numbers.",
              "<b>Class imbalance:</b> the classes are only mildly imbalanced. Class weights help recall on Fake, but the operating threshold (0.5) is not tuned.",
              "<b>Metadata shortcut and bias:</b> speaker and party fields can let the model learn who usually lies rather than what is false. "
              "This raises fairness concerns, and the model may not generalise to new speakers.",
              "<b>Overfitting and variance:</b> 110M parameters against 10k examples; results come from a single seed and 3 epochs. Several seeds, "
              "early stopping and hyper-parameter search would give more reliable estimates.",
              "<b>Temporal and domain shift:</b> the data covers US politics from 2007 to 2016. Performance on newer claims, other countries or social-media "
              "text is not tested.",
              "<b>Not a fact-checker:</b> the model scores how likely a statement is to be rated false. It does not retrieve or check evidence, "
              "so it must not be used as an automated verdict.",
              "<b>Possible improvements:</b> add the speaker's credit-history counts as numeric features, use evidence retrieval (as in LIAR-PLUS), "
              "try domain-adapted encoders (RoBERTa, DeBERTa), and calibrate the threshold."])

# ---- 8 Conclusion
s += [Paragraph("8. Conclusion", H1),
      Paragraph(f"A BERT-base classifier was fine-tuned for binary fake-news detection on LIAR. It reached {b.accuracy:.1%} test accuracy "
                f"and {b.f1_macro:.3f} macro-F1. That clearly beats the majority-class baseline and is on par with, or slightly better than, a TF-IDF + Logistic Regression model. "
                "The analysis shows the task is limited by the data, and points to evidence-aware models as the next step.", P)]

# ---- Appendix code
s += [PageBreak(), Paragraph("Appendix A — Complete Python Source Code", H1),
      Paragraph("The same code is provided as an executed Jupyter notebook (<i>Fake_News_BERT_23FE10ITE00008.ipynb</i>) and as "
                "<i>fake_news_bert.py</i>. Requirements: torch, transformers, scikit-learn, pandas, matplotlib, seaborn.", P), Spacer(1, 4)]
code = open("fake_news_bert.py").read()
for chunk in re.split(r"(?=^# %%)", code, flags=re.M):
    s.append(Preformatted(chunk.rstrip(), CODE, maxLineLength=135))

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Anwesha Mahesh Mulge · 23FE10ITE00008 · Case Study 13: Fake News Classification (BERT)")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

SimpleDocTemplate("Anwesha_Mulge_23FE10ITE00008_Fake_News_BERT_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=2 * cm, title="Fake News Classification using BERT — Anwesha Mulge").build(s, onFirstPage=footer, onLaterPages=footer)
print("PDF written")
