"""Build the project report .docx from the official IEEE template.

    python docs/report/build_docx.py

Why this exists
---------------
The report must follow IEEE conference format. The official Word template
(`conference-template-a4.docx`) supplies that format, but it cannot be edited
with python-docx: it is saved in ISO/IEC 29500 *Strict* OOXML, whose
relationship and markup namespaces live under `purl.oclc.org`, while
python-docx only understands the *Transitional* namespaces under
`schemas.openxmlformats.org`.

Rather than hand-editing the template's own document.xml -- which splits the
page into four sections (1, 3, 3 and 2 columns) to lay out the title and
author block -- this script keeps everything about the template *except* its
document.xml, and writes a new one that uses the template's own styles
(`papertitle`, `Author`, `Abstract`, `Keywords`, `Heading1`, `BodyText`,
`bulletlist`, `references`) and its page setup. The result is a standard
two-column IEEE layout with the title and authors spanning the full width.

The template's styles.xml, numbering.xml, theme, font table and settings are
carried across untouched, so no styling is reimplemented here.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
TEMPLATE = HERE / "conference-template-a4.docx"
OUTPUT = HERE / "report.docx"

W_NS = "http://purl.oclc.org/ooxml/wordprocessingml/main"


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
TITLE = (
    "Predicting House Prices from Exterior Photographs: Frozen Features, "
    "Partial Fine-Tuning, and What the Errors Reveal"
)

AUTHORS_LINE = "Mingyang Sun (1657392) and Yikai Qian (1722421)"
AFFILIATION_LINE = "The University of Melbourne"

ABSTRACT = (
    "Abstract—We predict the sale price of a house from a single exterior "
    "photograph, scored by mean squared error. Starting from a mean-price "
    "reference, we compare a frozen ImageNet ResNet-50 representation with "
    "partial fine-tuning and with data augmentation, then ensemble the "
    "fine-tuned runs. The final model reaches 96 959 on the test set, 15.1% "
    "below the reference. Our principal finding is negative and, we argue, "
    "more useful than a further accuracy gain: an error analysis shows that "
    "53% of the total squared error comes from the most expensive tenth of "
    "houses, which the ensemble under-predicts by around 650, while the "
    "cheapest three tenths are over-predicted by around 220. It also shows "
    "that ensembling improved the overall score while making that expensive "
    "decile worse. Several plausible improvements, including higher input "
    "resolution, a stronger backbone and an aspect-ratio correction, were "
    "tested and did not help; we report them alongside the ones that did."
)

KEYWORDS = (
    "Keywords—transfer learning, fine-tuning, regression, ensemble methods, "
    "error analysis"
)

# (style, text). "H1"/"H2" become Heading1/Heading2; "B" is a bullet.
BODY: list[tuple[str, str]] = [
    ("H1", "Introduction"),
    ("P", "The appearance of a home is widely believed to affect its sale "
          "price, and computer vision is increasingly used to quantify such "
          "effects. We take the task literally: given one photograph of a "
          "house exterior, predict its sale price in units of USD 1000, "
          "scored by mean squared error (MSE) against held-out listings."),
    ("P", "The task is difficult for a reason that shapes everything that "
          "follows. House price is dominated by location and floor area, "
          "neither of which is directly observable in a photograph; what a "
          "photograph offers is a proxy for building quality, apparent size "
          "and neighbourhood character. A model therefore operates with "
          "substantial irreducible uncertainty, and under MSE the optimal "
          "response to uncertainty is to stay close to the conditional mean. "
          "That mechanism, rather than any defect in the representation, "
          "turns out to explain most of our remaining error."),
    ("P", "We adopt a transfer-learning approach. An ImageNet-pretrained "
          "ResNet-50 supplies a general visual representation, and we compare "
          "three ways of using it: as a fixed feature extractor with a linear "
          "readout, as a base for partial fine-tuning, and with data "
          "augmentation applied during fine-tuning. We then average the "
          "fine-tuned runs. Our contributions are:"),
    ("B", "a controlled comparison of frozen versus fine-tuned "
          "representations on the same split, showing that partial "
          "fine-tuning is worth roughly five times as much as any "
          "input-resolution change we tried;"),
    ("B", "an ensemble that improves on the best single model by 4.6% on the "
          "test set, and the observation that this is the one change whose "
          "test score beat its validation estimate;"),
    ("B", "an error analysis that localises 53% of the total squared error "
          "in the most expensive tenth of houses, and shows that ensembling "
          "improves the overall score while making that decile worse; and"),
    ("B", "a set of honest negative results -- higher resolution, a stronger "
          "backbone, an aspect-ratio correction, and augmentation at a fixed "
          "epoch budget -- each of which we measured rather than assumed."),

    ("H1", "Method"),
    ("H2", "Data and split"),
    ("P", "The provided dataset contains 8000 labelled training images and "
          "3000 unlabelled test images, at a native resolution of about "
          "400 x 280 pixels. Prices range from 195 to 2000 (in USD 1000) with "
          "mean 720.7, median 639.9 and skewness 1.26; the distribution is "
          "markedly right-skewed."),
    ("P", "We hold out 1000 training images for validation and train on the "
          "remaining 7000. The split is stratified by price decile rather "
          "than drawn at random: with a skewed target, a plain random split "
          "leaves the expensive tail thinly represented in validation, which "
          "is precisely the region the error analysis needs to characterise. "
          "The split, the seed (42) and all subsequent preprocessing "
          "statistics are persisted so that every experiment reads the same "
          "partition, and all model selection is done on validation MSE. The "
          "test set is used only to produce final predictions."),
    ("H2", "Visual representation"),
    ("P", "We use an ImageNet-pretrained ResNet-50 [2] with the "
          "classification head removed. Images are resized to 224 x 224 and "
          "normalised with the network's own preprocessing; the output of the "
          "final convolutional stage is reduced by global average pooling [3] "
          "to a 2048-dimensional descriptor per image."),
    ("P", "Two readouts are compared on these features. The first is ridge "
          "regression, with the regularisation strength chosen by "
          "cross-validation on the training partition alone. The second is a "
          "small multilayer perceptron (256 hidden units, dropout 0.2) "
          "trained with Adam [4] and early stopping on validation loss. The "
          "target is standardised using training statistics in both cases, "
          "and predictions are clipped to the price range observed in "
          "training."),
    ("H2", "Partial fine-tuning"),
    ("P", "We then unfreeze the last residual stage (layer4, 14.9M of the "
          "backbone's 23.5M parameters) together with a new regression head, "
          "and train with Adam at 1e-4 and early stopping. Three variants are "
          "run: at 224 and 320 pixels, and with augmentation enabled."),
    ("P", "Augmentation is deliberately mild and targets the two directions "
          "in which the dataset genuinely varies. Horizontal flips, small "
          "translations and zooms address pose and framing; mild brightness "
          "and contrast jitter addresses lighting. Augmentation is applied to "
          "the training partition only."),
    ("H2", "Ensembling"),
    ("P", "Because the three fine-tuned runs differ in input resolution and "
          "in whether augmentation was used, they make partly different "
          "errors on the same images. We average their predictions -- a "
          "simple mean with no fitted weights, so that no additional "
          "selection is performed on the validation set."),

    ("H1", "Design Justification"),
    ("H2", "Why frozen features, and why partial fine-tuning"),
    ("P", "Convolutional networks learn a hierarchy in which early layers "
          "respond to generic structures such as edges, blobs and textures, "
          "while later layers respond to increasingly task-specific parts. "
          "Yosinski et al. [5] showed that these early features transfer "
          "between tasks while later ones are fragile, which is the standard "
          "argument for freezing most of a pretrained network. Real-estate "
          "exteriors differ from ImageNet's object-centric images in framing, "
          "lighting and scene composition, so some adaptation is warranted; "
          "but the case for adapting all of the network is weak, because our "
          "fine-tuned models reach a training loss of 0.068 against a "
          "validation loss of 0.631, a ratio of 0.108, indicating that "
          "capacity already exceeds what 7000 images can support. Unfreezing "
          "only the last stage is a compromise between the domain gap and "
          "that overfitting risk."),
    ("H2", "Why global average pooling"),
    ("P", "GAP makes the descriptor invariant to where a feature appears, "
          "which suits this task: a garage is a garage whether it sits left "
          "or right of frame. The cost is that all spatial layout is "
          "discarded, including the size of a building relative to its lot, a "
          "quantity that plausibly correlates with price. We note this as a "
          "limitation and return to it in the error analysis; alternative "
          "pooling strategies would be the natural next "
          "representation-level experiment."),
    ("H2", "Why a linear readout was competitive"),
    ("P", "Ridge outperformed the MLP on the frozen features (validation MSE "
          "114854 against 133379), and the MLP stopped after 19 epochs. "
          "Rather than reading this as evidence that prices are a linear "
          "function of the features, we read it as evidence about sample "
          "size: with 2048 features and 7000 examples, the non-linear head "
          "overfits before it can exploit its extra flexibility. This is "
          "consistent with the fine-tuning results, where unfreezing more "
          "parameters did not pay."),
    ("H2", "Why MSE on a skewed target is the central constraint"),
    ("P", "MSE is the task's specified metric, so it is not a free choice; "
          "but its consequences dominate our results. The squared loss "
          "penalises large deviations quadratically, so when a model is "
          "uncertain it minimises expected loss by predicting near the "
          "conditional mean. With a target of skewness 1.26, this pulls "
          "expensive houses downwards and cheap houses upwards. This is not a "
          "failure of fitting but the objective behaving as intended. Limited "
          "information in the photographs could produce a similar pattern, so "
          "we treat this as the explanation most consistent with our results "
          "rather than as the sole cause."),
    ("H2", "Why clipping, and why ensembling"),
    ("P", "A linear head extrapolates: on the test set, eight of 3000 "
          "predictions fell below the lowest price present in training. Since "
          "the target has a known empirical range, a prediction outside it is "
          "known to be wrong, and clipping to the training range can only "
          "help. Its measured effect on validation is small for ridge (about "
          "2 MSE) and larger for the MLP (about 680), which matches the "
          "intuition that the model with more freedom extrapolates further."),
    ("P", "Averaging the three fine-tuned runs reduces the variance "
          "component of the error while preserving the shared signal. Its "
          "effect is not uniform across the price range: it improves overall "
          "error by 3.2% relative to the best single model but makes the most "
          "expensive decile 4.3% worse, as the error analysis reports. The "
          "absolute Kaggle error is also 1.2% above the validation estimate, "
          "so the test score is not evidence that the ensemble generalises "
          "better than validation suggested; only the comparison against the "
          "single model favours it."),

    ("H1", "Data"),
    ("P", "[Section to be completed by Member A.]"),
    ("H1", "Experimental Setup"),
    ("P", "[Section to be completed by Member A.]"),
    ("H1", "Results"),
    ("P", "[Section to be completed by Member A. See docs/experiments.md for "
          "the comparison table and figures.]"),
    ("H1", "Error Analysis"),
    ("P", "We analysed the equal-weight E2/E3/E4 ensemble on the fixed "
          "1,000-image validation split. Predictions were matched by image "
          "identifier and retained at the stored one-decimal export "
          "precision. The ensemble achieved MSE 95,852.00, RMSE 309.60 and "
          "MAE 226.42, where prices and absolute errors are in USD 1,000. "
          "MSE was 3.23% lower than the best single model, E3, but absolute "
          "error improved on only 515 of the 1,000 images."),
    ("T", [
        ["True-price group", "n", "Mean signed error", "Share of squared error"],
        ["Lowest 30%", "300", "+222.03", "21.19%"],
        ["Middle 50%", "500", "+46.27", "16.81%"],
        ["Ninth decile", "100", "−179.81", "8.86%"],
        ["Highest 10%", "100", "−648.48", "53.14%"],
    ]),
    ("P", "Grouping images by true-price decile reveals systematic errors "
          "hidden by the small overall signed bias of +6.91. The "
          "lowest-price 30% were over-estimated by 222.03 on average. The "
          "highest-price 10% were under-estimated by 648.48 and contributed "
          "53.14% of total squared error. Their MSE was 509,349.70, 4.32% "
          "higher than E3 on the same images. Thus averaging improved overall "
          "performance while making the expensive tail less accurate. The "
          "lowest group MAE, 115.56, occurred between true prices of 639.45 "
          "and 718.93."),
    ("P", "We selected three examples each with the smallest absolute "
          "errors, the largest over-estimates and the largest "
          "under-estimates. 751.jpg was predicted at 758.2 against a true "
          "price of 759.0; 3947.jpg was over-estimated by 606.7 and 1977.jpg "
          "under-estimated by 1,449.1. These deliberately selected examples "
          "illustrate extremes and are not representative estimates of "
          "average performance."),
    ("P", "The group means show predictions concentrated near central "
          "prices. This may reflect limited information in exterior "
          "photographs, representation limits or fitting choices; the "
          "analysis does not establish a unique cause. Potential "
          "improvements include controlled comparisons of target "
          "transformations, sampling policies and spatial representations, "
          "evaluated using original-unit MSE. These findings are descriptive "
          "results from one validation split already used for model "
          "development, not independent evidence of generalisation."),

    ("H1", "Conclusion"),
    ("P", "We predicted house sale prices from exterior photographs and "
          "improved on a mean-price reference by 15.1% on the test set. The "
          "gain came from two places: partially fine-tuning an ImageNet "
          "ResNet-50, worth about five times as much as any input-resolution "
          "change we tried, and averaging three fine-tuned runs, which "
          "reduced error by a further 4.6%."),
    ("P", "What we did not find is equally part of the result. Higher input "
          "resolution, a stronger backbone, an aspect-ratio correction and "
          "augmentation at a fixed epoch budget were each tested in a "
          "controlled comparison and each failed to help, in several cases "
          "for reasons we can name. Reporting these matters because the "
          "alternative -- trying something until the leaderboard moves -- "
          "would have produced a weaker account of why the final model "
          "works."),
    ("P", "The error analysis points to the clearest direction for future "
          "work, and it is not an architectural one. 53% of the total squared "
          "error comes from the most expensive tenth of houses, which the "
          "ensemble under-predicts by 648 on average, while the cheapest "
          "three tenths are over-predicted by 222. Ensembling improved the "
          "overall score while making that decile worse, which suggests that "
          "further gains need to target the tail rather than add capacity. "
          "The pattern is consistent with squared-error optimisation on a "
          "right-skewed target, so the natural next steps change the "
          "objective rather than the network: "
          "predicting the logarithm of price, which compresses the tail the "
          "squared loss is most sensitive to; quantile or ordinal regression, "
          "which does not reward retreating to the mean; or weighting "
          "expensive samples more heavily during training. A second direction "
          "is representational, since global average pooling discards the "
          "spatial layout needed to judge a building's size relative to its "
          "lot."),
    ("P", "More generally, this study is a reminder that on a task where the "
          "input carries limited information about the target, careful "
          "measurement of where a model fails is worth more than further "
          "tuning of a model that is already close to what its inputs allow."),
]

REFERENCES = [
    "T. Dogan, “House Prices and Images -- SoCal,” Kaggle, 2019. "
    "[Online]. Available: https://www.kaggle.com/datasets/ted8080/"
    "house-prices-and-images-socal",

    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for "
    "image recognition,” in Proc. IEEE Conf. Computer Vision and Pattern "
    "Recognition (CVPR), 2016, pp. 770–778.",

    "M. Lin, Q. Chen, and S. Yan, “Network in network,” in Proc. Int. "
    "Conf. Learning Representations (ICLR), 2014.",

    "D. P. Kingma and J. Ba, “Adam: A method for stochastic "
    "optimization,” in Proc. Int. Conf. Learning Representations (ICLR), "
    "2015.",

    "J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable "
    "are features in deep neural networks?” in Advances in Neural "
    "Information Processing Systems (NeurIPS), 2014, pp. 3320–3328.",
]

STYLE_FOR = {"P": "BodyText", "B": "bulletlist", "H1": "Heading1", "H2": "Heading2"}


# --------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def para(
    style: str,
    text: str,
    trailing_sectpr: str = "",
    spacing: str = "",
    sz: int | None = None,
) -> str:
    """One <w:p>.

    ``spacing`` and ``sz`` apply direct formatting over the named style.
    They exist because the template's ``Author`` style carries 18pt of space
    before every paragraph, which costs about 72pt across the author block --
    roughly a tenth of a page, for text that should be compact.
    ``sz`` is in half-points, as OOXML requires.
    """
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/>{spacing}{trailing_sectpr}</w:pPr>'
    rpr = f'<w:rPr><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>' if sz else ""
    body = (
        f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>' if text else ""
    )
    return f"<w:p>{ppr}{body}</w:p>"


def table(rows: list[list[str]]) -> str:
    """A compact table sized to the column, with hairline rules.

    The template defines no general-purpose table style (only header/caption
    styles for its own examples), so borders and widths are set explicitly
    rather than relying on a style that may not exist. Row 0 is the header and
    is bold.
    """
    ncols = max(len(r) for r in rows)
    width = 4800 // ncols  # twips; the two-column text area is ~243pt wide
    borders = (
        "<w:tblBorders>"
        + "".join(
            f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + "</w:tblBorders>"
    )
    out = [
        "<w:tbl><w:tblPr>"
        '<w:tblW w:w="5000" w:type="pct"/>'
        + borders
        + '<w:tblLayout w:type="fixed"/>'
        "</w:tblPr><w:tblGrid>"
        + "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(ncols))
        + "</w:tblGrid>"
    ]
    for i, row in enumerate(rows):
        cells = list(row) + [""] * (ncols - len(row))
        out.append("<w:tr>")
        for cell in cells:
            bold = "<w:b/>" if i == 0 else ""
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="BodyText"/>'
                f'<w:spacing w:before="0" w:after="0"/><w:jc w:val="left"/></w:pPr>'
                f'<w:r><w:rPr>{bold}<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(cell)}</w:t></w:r></w:p></w:tc>'
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    # Word needs a paragraph after a table so the following one is not merged.
    out.append('<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p>')
    return "".join(out)


def build_body(root_attrs: str, title_sectpr: str, body_sectpr: str) -> str:
    out: list[str] = []

    out.append(para("papertitle", TITLE))
    out.append(
        para(
            "Author",
            AUTHORS_LINE,
            spacing='<w:spacing w:before="8pt" w:after="2pt"/>',
            sz=22,  # 11pt
        )
    )
    out.append(
        para(
            "Affiliation",
            AFFILIATION_LINE,
            spacing='<w:spacing w:before="0" w:after="0"/>',
            sz=18,  # 9pt
        )
    )
    # Section break that closes the single-column title/author block.
    out.append(f'<w:p><w:pPr>{title_sectpr}</w:pPr></w:p>')

    out.append(para("Abstract", ABSTRACT))
    out.append(para("Keywords", KEYWORDS))

    for style, text in BODY:
        if style == "T":
            out.append(table(text))
        else:
            out.append(para(STYLE_FOR[style], text))

    for ref in REFERENCES:
        out.append(para("references", ref))

    return (
        f"<w:document{root_attrs}><w:body>"
        + "".join(out)
        + body_sectpr
        + "</w:body></w:document>"
    )


def main() -> None:
    src = zipfile.ZipFile(TEMPLATE)
    original = src.read("word/document.xml").decode("utf-8")

    # Reuse the template's own namespace declarations verbatim.
    root_attrs = re.search(r"<w:document([^>]*)>", original).group(1)

    # Title block: one column. Body: the template's two-column page setup.
    two_col = re.findall(r"<w:sectPr.*?</w:sectPr>", original, re.S)
    if not any('w:num="2"' in s for s in two_col):
        raise RuntimeError("template has no two-column section to copy")
    body_sectpr = next(s for s in two_col if 'w:num="2"' in s)

    one_col = re.sub(r"<w:cols[^>]*/>", '<w:cols w:num="1" w:space="36pt"/>', body_sectpr)
    title_sectpr = one_col

    new_doc = build_body(root_attrs, title_sectpr, body_sectpr)

    # Copy every part of the template except document.xml.
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = new_doc.encode("utf-8")
            dst.writestr(item, data)
    src.close()

    print(f"wrote {OUTPUT}")
    print(f"  paragraphs: {new_doc.count('<w:p>') + new_doc.count('<w:p ')}")
    print(f"  two-column body section: {'w:num=\"2\"' in new_doc}")
    print(f"  styles used: papertitle, Author, Abstract, Keywords, "
          f"{', '.join(sorted(set(STYLE_FOR.values())))}")


if __name__ == "__main__":
    main()
