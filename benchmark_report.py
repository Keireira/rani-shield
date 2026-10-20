"""CPU-only paired reporting for benchmark_chain / benchmark_suite outputs.

API: summarize(pair_reports, epsilon=1e-6, minimum_baseline=.80) -> dict.
     write_report(out_dir, pair_reports, metadata=None, *, root=None,
                  contact_pairs=6, minimum_baseline=.80) -> paths plus summary.
No network, Torch, model loading, or GPU operations are used.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


METHODS = ("clean", "stage1", "final")
LABELS = {"clean": "Clean swap", "stage1": "PhantomSeal swap", "final": "PhantomSeal + FaceShield swap"}
LIMITATIONS = [
    "Lower FaceNet cosine means the swap is less similar to the CLEAN source identity. It is a relative identity-copying metric, not a calibrated success probability.",
    "Primary numerical comparisons require a complete paired triple, successful source/output face detection, finite scores, and a clean-swap cosine at or above the configured minimum baseline. All unfiltered numerical values and images remain available.",
    "The default 0.80 baseline quality gate is a user-requested heuristic, not a validated universal FaceNet identity threshold or a probability. Passing it only qualifies a pair for exploratory numerical comparison; manual baseline inspection is still required before any protection conclusion.",
    "The primary aggregate delta averages target-level differences within each source, then gives each source equal weight. Pair-level means and counts are descriptive; repeated targets are not independent source identities.",
    "FaceNet was used for cloak selection, so it is not an entirely unseen evaluator. SimSwap was an optimization branch of PhantomSeal, so this is not an unseen-swapper transfer test.",
    "One swapper and a small public sample cannot establish general protection, robustness to image processing, or protection against other tools. No p-values or thresholded attack-success rates are reported.",
    "A negative final-minus-PhantomSeal delta favors the chain for this metric; a positive delta favors PhantomSeal alone. Detection failures are separate outcomes, not invented zero similarity scores.",
    "PNG and JPEG95 conditions are reported separately because processing can reverse their ordering. All similarities should use the original clean PNG identity reference; the clean swap baseline itself uses the same processing condition as the protected sources.",
    "The additional source-versus-target baseline sanity check is reported separately. The configured clean-source quality gate controls primary numerical inclusion; neither check substitutes for visually confirming a recognizable clean swap.",
    "This evaluates protected photos in the source-identity role: the attacker extracts their identity for insertion into another face. It does not test preventing replacement of the protected photo's face when that photo is the destination.",
    "PhantomSeal alone and the chain are compared at their current product settings, not at matched perturbation budgets. Any additional protection may come with greater image distortion; this is not a matched-budget ablation.",
]


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _direction(delta, epsilon):
    if delta < -epsilon:
        return "chain_better"
    if delta > epsilon:
        return "ps_better"
    return "ties"


def _pair_ids(report):
    if not isinstance(report, Mapping):
        raise ValueError("Every pair report must be an object.")
    values = []
    for name in ("source_id", "target_id", "condition"):
        value = report.get(name, "png" if name == "condition" else None)
        if value is None or isinstance(value, (dict, list, bool)) or not str(value).strip():
            raise ValueError(f"Every report requires a nonempty {name}.")
        if isinstance(value, Real) and not math.isfinite(float(value)):
            raise ValueError(f"{name} cannot be non-finite.")
        values.append(str(value))
    return tuple(values)


def _analyse(report, minimum_baseline=.80):
    source, target, condition = _pair_ids(report)
    raw_rows = report.get("rows", {})
    raw_rows = raw_rows if isinstance(raw_rows, Mapping) else {}
    rows, reasons, scores = {}, {}, {}
    for method in METHODS:
        raw = raw_rows.get(method)
        row = raw if isinstance(raw, Mapping) else {}
        rows[method] = row
        problems = []
        if not isinstance(raw, Mapping):
            problems.append("missing_row")
        for key in ("source_face_detected", "swap_face_detected"):
            if row.get(key) is not True:
                problems.append(key + ("_false" if row.get(key) is False else "_missing_or_invalid"))
        score = row.get("facenet_cosine_to_clean_source")
        if score is None:
            problems.append("score_null_or_missing")
        elif not _finite_number(score):
            problems.append("score_nonfinite_or_invalid")
        reasons[method] = problems
        scores[method] = float(score) if _finite_number(score) else None
    complete = not any(reasons.values())
    clean_target = rows["clean"].get("facenet_cosine_to_target")
    sanity_available = not reasons["clean"] and _finite_number(clean_target)
    sanity = ("pass" if scores["clean"] > float(clean_target) else "fail") if sanity_available else "unavailable"
    gate = ("pass" if scores["clean"] >= minimum_baseline else "weak") if not reasons["clean"] else "unavailable"
    return {"source_id": source, "target_id": target, "condition": condition, "rows": rows,
            "reasons": reasons, "scores": scores, "complete": complete,
            "qualified": complete and gate == "pass", "baseline_gate": gate,
            "delta": scores["final"] - scores["stage1"] if complete else None,
            "baseline_sanity": sanity}


def _analyses(reports, minimum_baseline=.80):
    result, seen = [], set()
    for report in reports:
        pair = _analyse(report, minimum_baseline)
        key = (pair["source_id"], pair["target_id"], pair["condition"])
        if key in seen:
            raise ValueError(f"Duplicate source/target pair: {key}")
        seen.add(key)
        result.append(pair)
    return result


def _numerical_aggregates(pairs, included, epsilon):
    """Identical paired/source weighting for the primary and raw populations."""
    by_source = defaultdict(list)
    for pair in pairs:
        by_source[pair["source_id"]].append(pair)
    selected = defaultdict(list)
    for pair in included:
        selected[pair["source_id"]].append(pair)
    counts = {name: 0 for name in ("chain_better", "ps_better", "ties")}
    source_counts = counts.copy()
    for pair in included:
        counts[_direction(pair["delta"], epsilon)] += 1
    per_source = []
    for source_id in sorted(by_source):
        valid = selected[source_id]
        means = {method: _mean([pair["scores"][method] for pair in valid]) for method in METHODS}
        delta = _mean([pair["delta"] for pair in valid])
        direction = _direction(delta, epsilon) if delta is not None else None
        if direction:
            source_counts[direction] += 1
        per_source.append({"source_id": source_id, "total_pairs": len(by_source[source_id]),
                           "included_pairs": len(valid), "mean_scores": means,
                           "mean_delta_final_minus_stage1": delta, "direction": direction})
    valid_sources = [source for source in per_source if source["included_pairs"]]
    return {"included_pairs": len(included), "sources_with_included_pairs": len(valid_sources),
        "mean_scores": {method: _mean([pair["scores"][method] for pair in included]) for method in METHODS},
        "mean_delta_final_minus_stage1": _mean([pair["delta"] for pair in included]),
        "source_weighted_mean_scores": {method: _mean([source["mean_scores"][method] for source in valid_sources]) for method in METHODS},
        "source_weighted_delta_final_minus_stage1": _mean([source["mean_delta_final_minus_stage1"] for source in valid_sources]),
        "pair_comparison_counts": counts, "source_comparison_counts": source_counts, "per_source": per_source}


def _summarize(pair_reports, epsilon, include_conditions, minimum_baseline):
    if not _finite_number(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite nonnegative number.")
    if not _finite_number(minimum_baseline) or not 0 <= minimum_baseline <= 1:
        raise ValueError("minimum_baseline must be a finite number between 0 and 1.")
    reports = list(pair_reports)
    pairs = _analyses(reports, minimum_baseline)
    complete = [pair for pair in pairs if pair["complete"]]
    qualified = [pair for pair in pairs if pair["qualified"]]
    primary = _numerical_aggregates(pairs, qualified, epsilon)
    raw = _numerical_aggregates(pairs, complete, epsilon)
    raw["complete_pairs"] = len(complete)
    raw["excluded_pairs"] = len(pairs) - len(complete)
    raw["note"] = "Unfiltered descriptive numerical statistics, including weak clean baselines; not evidence of protection efficacy."
    failures = {
        method: {"source_detection_failed": 0, "swap_detection_failed": 0,
                 "swap_not_attempted_after_source_failure": 0,
                 "missing_or_invalid_detection_flags": 0, "either_detection_not_successful": 0}
        for method in METHODS
    }
    invalid_scores = {method: 0 for method in METHODS}
    invalid_methods = {method: 0 for method in METHODS}
    for pair in pairs:
        for method in METHODS:
            row = pair["rows"][method]
            source, swap = row.get("source_face_detected"), row.get("swap_face_detected")
            failures[method]["source_detection_failed"] += source is False
            failures[method]["swap_detection_failed"] += source is True and swap is False
            failures[method]["swap_not_attempted_after_source_failure"] += source is False and swap is False
            failures[method]["missing_or_invalid_detection_flags"] += not isinstance(source, bool) or not isinstance(swap, bool)
            failures[method]["either_detection_not_successful"] += source is not True or swap is not True
            invalid_scores[method] += pair["scores"][method] is None
            invalid_methods[method] += bool(pair["reasons"][method])
    for source in primary["per_source"]:
        source_pairs = [pair for pair in pairs if pair["source_id"] == source["source_id"]]
        source["complete_pairs"] = sum(pair["complete"] for pair in source_pairs)
        source["qualified_pairs"] = source["included_pairs"]
        source["weak_baseline_pairs"] = sum(pair["complete"] and pair["baseline_gate"] == "weak" for pair in source_pairs)
        source["excluded_pairs"] = source["total_pairs"] - source["qualified_pairs"]
    for source in raw["per_source"]:
        source["complete_pairs"] = source["included_pairs"]
    result = {
        **primary,
        "total_pairs": len(pairs), "complete_pairs": len(complete), "qualified_pairs": len(qualified),
        "excluded_pairs": len(pairs) - len(qualified), "incomplete_pairs": len(pairs) - len(complete),
        "weak_baseline_pairs": sum(pair["complete"] and pair["baseline_gate"] == "weak" for pair in pairs),
        "total_sources": len(primary["per_source"]), "sources_with_complete_pairs": raw["sources_with_included_pairs"],
        "sources_with_qualified_pairs": primary["sources_with_included_pairs"],
        "minimum_baseline": float(minimum_baseline), "manual_baseline_inspection_required": True,
        "baseline_gate_note": "User-requested heuristic quality gate; not a validated universal FaceNet threshold. Qualified comparisons remain exploratory until manual baseline inspection.",
        "baseline_gate_counts": {status: sum(pair["baseline_gate"] == status for pair in pairs)
                                 for status in ("pass", "weak", "unavailable")},
        "raw": raw,
        "tie_epsilon": float(epsilon), "detection_failures": failures,
        "invalid_scores_by_method": invalid_scores, "invalid_methods_by_method": invalid_methods,
        "baseline_sanity_checks": {"available_pairs": sum(pair["baseline_sanity"] != "unavailable" for pair in pairs),
                                   "passed": sum(pair["baseline_sanity"] == "pass" for pair in pairs),
                                   "failed": sum(pair["baseline_sanity"] == "fail" for pair in pairs),
                                   "unavailable": sum(pair["baseline_sanity"] == "unavailable" for pair in pairs)},
        "excluded_pair_details": [{"source_id": pair["source_id"], "target_id": pair["target_id"], "condition": pair["condition"],
                                    "baseline_gate": pair["baseline_gate"],
                                    "reasons": {**{method: reasons for method, reasons in pair["reasons"].items() if reasons},
                                                **({"baseline": ["weak_clean_baseline"]} if pair["baseline_gate"] == "weak" else {})}}
                                   for pair in pairs if not pair["qualified"]],
        "limitations": LIMITATIONS.copy(),
    }
    if include_conditions:
        by_condition = defaultdict(list)
        for report, pair in zip(reports, pairs):
            by_condition[pair["condition"]].append(report)
        result["by_condition"] = {name: _summarize(group, epsilon, False, minimum_baseline)
                                  for name, group in sorted(by_condition.items())}
    return result


def summarize(pair_reports, epsilon=1e-6, minimum_baseline=.80):
    """Primary baseline-qualified triples plus preserved raw numerical statistics."""
    return _summarize(pair_reports, epsilon, True, minimum_baseline)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported report metadata type: {type(value).__name__}")


def _formatted(value):
    return f"{value:.4f}" if _finite_number(value) else "n/a"


def _representatives(pairs, count):
    ordered = sorted(range(len(pairs)), key=lambda i: (pairs[i]["source_id"], pairs[i]["target_id"], pairs[i]["condition"]))
    count = min(max(0, int(count)), len(ordered))
    if count == 0:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)]


def _font(size=16):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _quality_html(metadata):
    values = metadata.get("image_quality", [])
    if not isinstance(values, list) or not values:
        return ""
    rows = []
    for source in values:
        if not isinstance(source, Mapping):
            continue
        for method in ("stage1", "final"):
            metrics = source.get(method, {})
            if not isinstance(metrics, Mapping):
                continue
            label = "PhantomSeal" if method == "stage1" else "PhantomSeal + FaceShield"
            rows.append("<tr><td>" + html.escape(str(source.get("source_id", "unknown"))) + "</td><td>" + label + "</td>"
                        + "".join("<td>" + _formatted(metrics.get(key)) + "</td>" for key in ("ssim", "psnr_db", "mae", "max_pixel_change")) + "</tr>")
    return '''<div class="panel"><h2>Protected-image quality at current product settings</h2>
<p>These source-image quality metrics compare each protected photo with its clean baseline. PhantomSeal alone and the chain can use different total perturbation budgets: stronger effects, if observed, may cost greater distortion. This is not a matched-budget ablation.</p>
<div class="scroll"><table><thead><tr><th>Source</th><th>Method</th><th>SSIM</th><th>PSNR (dB)</th><th>MAE</th><th>Maximum pixel change</th></tr></thead><tbody>''' + "".join(rows) + "</tbody></table></div></div>"


def _contact_sheet(path, selected, pairs, previews):
    if not selected:
        return None
    cell, gap, row_extra, heading = 224, 12, 62, 82
    columns = ("source", "target", "clean", "stage1", "final")
    titles = ("Clean source", "Fixed target", "Clean swap", "PhantomSeal", "PS + FaceShield")
    sheet = Image.new("RGB", (gap + len(columns) * (cell + gap), heading + len(selected) * (cell + row_extra)), "#f3f5f7")
    draw, font, small = ImageDraw.Draw(sheet), _font(17), _font(14)
    draw.text((gap, 8), "Evenly spaced source/target/condition pairs; selected without looking at scores", fill="#17212c", font=font)
    for column, title in enumerate(titles):
        draw.text((gap + column * (cell + gap), 43), title, fill="#17212c", font=font)
    for row_index, pair_index in enumerate(selected):
        pair, y = pairs[pair_index], heading + row_index * (cell + row_extra)
        for column, kind in enumerate(columns):
            x = gap + column * (cell + gap)
            draw.rectangle((x, y, x + cell, y + cell), fill="white")
            preview = previews.get((pair_index, kind))
            if preview:
                with Image.open(preview) as opened:
                    thumbnail = ImageOps.contain(opened.convert("RGB"), (cell, cell), Image.Resampling.LANCZOS)
                sheet.paste(thumbnail, (x + (cell - thumbnail.width) // 2, y + (cell - thumbnail.height) // 2))
            else:
                draw.text((x + 8, y + 90), "Image unavailable", fill="#6b7280", font=small)
            if kind in METHODS:
                score = pair["scores"][kind] if not pair["reasons"][kind] else None
                draw.text((x, y + cell + 5), "cosine " + _formatted(score), fill="#17212c", font=small)
        label = f"Source {pair['source_id']} / target {pair['target_id']} / {pair['condition']}"
        draw.text((gap, y + cell + 27), label[:115], fill="#17212c", font=small)
    sheet.save(path)
    return str(path)


def write_report(out_dir, pair_reports, metadata=None, *, root=None, contact_pairs=6, minimum_baseline=.80):
    """Write portable JSON/CSV/HTML and score-agnostic contact-sheet previews.

    out_dir is a directory. Relative input image paths resolve against root
    (default cwd); thumbnails are copied under out_dir/assets for portability.
    Returns summary and absolute output paths. Existing source images are read
    only; previews may be resized, and benchmark scores are never recomputed.
    """
    reports = list(pair_reports)
    metadata = metadata or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object.")
    summary, pairs = summarize(reports, minimum_baseline=minimum_baseline), _analyses(reports, minimum_baseline)
    output = Path(out_dir).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root is not None else Path.cwd()
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    previews, preview_errors, copied = {}, [], {}
    for index, (pair, report) in enumerate(zip(pairs, reports)):
        clean_row = pair["rows"]["clean"]
        target_info = report.get("target", {})
        target_info = target_info if isinstance(target_info, Mapping) else {}
        values = {"source": clean_row.get("source_path"), "target": target_info.get("path")}
        values.update({method: pair["rows"][method].get("output_path") for method in METHODS})
        for kind, value in values.items():
            if not isinstance(value, (str, Path)) or not str(value):
                continue
            text_path = str(value)
            if "://" in text_path or text_path.startswith("data:"):
                preview_errors.append({"pair_index": index, "kind": kind, "error": "Only local image paths are accepted"})
                continue
            image_path = Path(value)
            image_path = image_path if image_path.is_absolute() else base / image_path
            image_path = image_path.resolve()
            if image_path in copied:
                previews[(index, kind)] = copied[image_path]
                continue
            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                preview_path = assets / f"pair_{index:03d}_{kind}.png"
                image.save(preview_path)
                copied[image_path] = preview_path
                previews[(index, kind)] = preview_path
            except (OSError, ValueError) as error:
                preview_errors.append({"pair_index": index, "kind": kind, "error": str(error)})
    selected = _representatives(pairs, contact_pairs)
    contact_path = output / "contact_sheet.png"
    contact_result = _contact_sheet(contact_path, selected, pairs, previews)
    document = {"schema_version": 2, "summary": summary, "metadata": metadata or {}, "pair_reports": reports,
                "representative_selection": "Evenly spaced in sorted source/target/condition ID order; no score-based selection",
                "representative_pairs": [{"source_id": pairs[i]["source_id"], "target_id": pairs[i]["target_id"], "condition": pairs[i]["condition"]} for i in selected],
                "preview_errors": preview_errors}
    json_path = output / "report.json"
    json_path.write_text(json.dumps(_json_safe(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = output / "pairs.csv"
    fields = ["source_id", "target_id", "condition", "complete_triple", "qualified_comparison",
              "minimum_baseline", "baseline_gate", "baseline_sanity"]
    for method in METHODS:
        fields.extend([f"{method}_cosine", f"{method}_source_detected", f"{method}_swap_detected"])
    fields.extend(["final_minus_stage1", "raw_final_minus_stage1", "direction", "exclusion_reasons"])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {"source_id": pair["source_id"], "target_id": pair["target_id"], "condition": pair["condition"],
                   "complete_triple": pair["complete"], "qualified_comparison": pair["qualified"],
                   "minimum_baseline": summary["minimum_baseline"], "baseline_gate": pair["baseline_gate"],
                   "baseline_sanity": pair["baseline_sanity"],
                   "final_minus_stage1": pair["delta"] if pair["qualified"] else None,
                   "raw_final_minus_stage1": pair["delta"],
                   "direction": (_direction(pair["delta"], summary["tie_epsilon"]) if pair["qualified"]
                                 else "inconclusive_weak_baseline" if pair["baseline_gate"] == "weak" else "inconclusive_invalid_triple"),
                   "exclusion_reasons": json.dumps({**{key: value for key, value in pair["reasons"].items() if value},
                                                    **({"baseline": ["weak_clean_baseline"]} if pair["baseline_gate"] == "weak" else {})})}
            for method in METHODS:
                row[f"{method}_cosine"] = pair["scores"][method]
                row[f"{method}_source_detected"] = pair["rows"][method].get("source_face_detected")
                row[f"{method}_swap_detected"] = pair["rows"][method].get("swap_face_detected")
            writer.writerow(row)

    escape = lambda value: html.escape(str(value), quote=True)
    def image_tag(index, kind):
        path = previews.get((index, kind))
        if not path:
            return '<span class="muted">Image unavailable</span>'
        relative = path.relative_to(output).as_posix()
        return f'<a href="{escape(relative)}"><img src="{escape(relative)}" alt="{escape(kind)}" loading="lazy"></a>'
    table_rows = []
    for index, pair in enumerate(pairs):
        cells = [f"<td>{escape(pair['source_id'])}<br><span class=muted>{escape(pair['target_id'])}<br>{escape(pair['condition'])}</span></td>",
                 f"<td>{image_tag(index, 'source')}</td>", f"<td>{image_tag(index, 'target')}</td>"]
        for method in METHODS:
            score = pair["scores"][method] if not pair["reasons"][method] else None
            detection_text = f"source={pair['rows'][method].get('source_face_detected')} / swap={pair['rows'][method].get('swap_face_detected')}"
            cells.append(f"<td>{image_tag(index, method)}<br><strong>{_formatted(score)}</strong><br><small>{escape(detection_text)}</small></td>")
        status = (_direction(pair["delta"], summary["tie_epsilon"]) if pair["qualified"] else
                  "Inconclusive — weak baseline" if pair["baseline_gate"] == "weak" else
                  "Inconclusive — missing or invalid scores")
        reasons = "; ".join(f"{method}: {', '.join(value)}" for method, value in pair["reasons"].items() if value)
        effect = pair["delta"] if pair["qualified"] else None
        cells.append(f"<td><strong>{_formatted(effect)}</strong><br>{escape(status)}<br><small>baseline gate: {escape(pair['baseline_gate'])}<br>baseline sanity: {escape(pair['baseline_sanity'])}<br>{escape(reasons)}</small></td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    source_rows = "".join(
        "<tr><td>" + escape(source["source_id"]) + "</td><td>" + str(source["qualified_pairs"]) + "/" + str(source["complete_pairs"]) + "/" + str(source["total_pairs"]) + "</td>"
        + "".join("<td>" + _formatted(source["mean_scores"][method]) + "</td>" for method in METHODS)
        + "<td>" + _formatted(source["mean_delta_final_minus_stage1"]) + "</td></tr>" for source in summary["per_source"])
    limitation_html = "".join("<li>" + escape(note) + "</li>" for note in LIMITATIONS)
    means = summary["source_weighted_mean_scores"]
    counts = summary["pair_comparison_counts"]
    quality_html = _quality_html(metadata)
    heldout_parts = []
    if metadata.get("reuses_exploratory_source") is True:
        heldout_parts.append("This sample reuses a source from the earlier exploratory portrait/cage case and is not a strictly held-out source evaluation.")
    if metadata.get("exploratory_source_ids"):
        heldout_parts.append("Exploratory source IDs: " + ", ".join(str(value) for value in metadata["exploratory_source_ids"]))
    if metadata.get("heldout_note"):
        heldout_parts.append(str(metadata["heldout_note"]))
    heldout_html = '<p><strong>Sample reuse / held-out status:</strong> ' + escape(" ".join(heldout_parts)) + '</p>' if heldout_parts else ""
    correction_html = ('<div class="panel"><strong>Correction:</strong> ' + escape(metadata["correction_note"]) + '</div>') if metadata.get("correction_note") else ""
    condition_rows = []
    for condition, group in summary["by_condition"].items():
        group_means = group["source_weighted_mean_scores"]
        condition_rows.append("<tr><td>" + escape(condition) + "</td><td>" + str(group["qualified_pairs"]) + "/" + str(group["complete_pairs"]) + "/" + str(group["total_pairs"]) + "</td>"
                              + "".join("<td>" + _formatted(group_means[method]) + "</td>" for method in METHODS)
                              + "<td>" + _formatted(group["source_weighted_delta_final_minus_stage1"]) + "</td><td>"
                              + str(group["weak_baseline_pairs"]) + "</td><td>" + str(group["baseline_sanity_checks"]["failed"]) + "</td></tr>")
    failures_html = "".join(f"<tr><td>{escape(LABELS[method])}</td><td>{values['source_detection_failed']}</td><td>{values['swap_detection_failed']}</td><td>{values['swap_not_attempted_after_source_failure']}</td><td>{values['missing_or_invalid_detection_flags']}</td><td>{summary['invalid_scores_by_method'][method]}</td></tr>" for method, values in summary["detection_failures"].items())
    contact_html = '<p><a href="contact_sheet.png">Open contact sheet</a></p><img class="sheet" src="contact_sheet.png" alt="Score-independent representative pairs">' if contact_result else ""
    raw = summary["raw"]
    raw_means = raw["source_weighted_mean_scores"]
    raw_counts = raw["pair_comparison_counts"]
    raw_condition_rows = "".join("<tr><td>" + escape(condition) + "</td><td>" + str(group["raw"]["complete_pairs"]) + "</td>"
        + "".join("<td>" + _formatted(group["raw"]["source_weighted_mean_scores"][method]) + "</td>" for method in METHODS)
        + "<td>" + _formatted(group["raw"]["source_weighted_delta_final_minus_stage1"]) + "</td></tr>"
        for condition, group in summary["by_condition"].items())
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SimSwap paired identity benchmark</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;color:#17212c;background:#f4f6f8;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}h1,h2{{line-height:1.2}}.panel{{background:white;border:1px solid #dce1e7;border-radius:10px;padding:18px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce1e7}}th{{background:#eef2f6}}td img{{width:135px;height:135px;object-fit:contain;background:#f2f3f5}}small,.muted{{font-size:12px;color:#586576}}.scroll{{overflow:auto}}.sheet{{max-width:100%;height:auto}}code{{font-size:13px}}a{{color:#1259a3}}</style>
<main><h1>SimSwap paired identity benchmark</h1><p><strong>Protected role: source identity.</strong> The attacker extracts identity from the protected photo and inserts it into another face. This does not test preventing face replacement when the protected photo is the destination.</p><p>Clean original → PhantomSeal → PhantomSeal + FaceShield. Lower cosine means less similarity to the clean source identity.</p>{heldout_html}
{correction_html}<div class="panel"><strong>{summary['qualified_pairs']} / {summary['total_pairs']} baseline-qualified numerical triples</strong>; {summary['sources_with_qualified_pairs']} / {summary['total_sources']} sources contribute.<br>
Clean baseline cosine must be ≥ {_formatted(summary['minimum_baseline'])}. This is a user-requested heuristic quality gate, not a validated universal FaceNet threshold or probability.<br>
<strong>Exploratory numerical comparison only — manual baseline inspection required.</strong> Passing the gate does not confirm a recognizable clean swap or establish protection efficacy.<br>
Qualified equal-source mean cosine: clean {_formatted(means['clean'])}; PhantomSeal {_formatted(means['stage1'])}; chain {_formatted(means['final'])}.<br>
<strong>Qualified equal-source final − PhantomSeal delta: {_formatted(summary['source_weighted_delta_final_minus_stage1'])}</strong> (numerical difference only).<br>
Qualified pair counts: chain lower {counts['chain_better']}; PhantomSeal lower {counts['ps_better']}; ties {counts['ties']} (epsilon {summary['tie_epsilon']:g}). Inconclusive weak baseline: {summary['weak_baseline_pairs']}; incomplete triples: {summary['incomplete_pairs']}.<br>
<a href="report.json">Full JSON</a> · <a href="pairs.csv">Pair CSV</a></div>
<div class="panel"><h2>Scope and interpretation</h2><ul>{limitation_html}</ul></div>
{quality_html}
<div class="panel"><h2>Separate processing conditions</h2><p>Primary means use only baseline-qualified triples and give each contributing source equal weight. Manual baseline inspection is still required.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Qualified / complete / total</th><th>Clean swap</th><th>PhantomSeal</th><th>Chain</th><th>Final − PS</th><th>Weak baseline excluded</th><th>Baseline sanity failed</th></tr></thead><tbody>{''.join(condition_rows)}</tbody></table></div></div>
<div class="panel"><h2>Per-source means on baseline-qualified triples</h2><div class="scroll"><table><thead><tr><th>Source</th><th>Qualified / complete / total</th><th>Clean swap</th><th>PhantomSeal</th><th>Chain</th><th>Final − PS</th></tr></thead><tbody>{source_rows}</tbody></table></div></div>
<div class="panel"><details><summary>Raw unfiltered numerical statistics — not an efficacy conclusion</summary><p>All {raw['complete_pairs']} complete triples remain available, including weak clean baselines. These unfiltered values cannot establish whether either defense worked. No cases, scores or images were removed.</p><p>Raw equal-source means: clean {_formatted(raw_means['clean'])}; PhantomSeal {_formatted(raw_means['stage1'])}; chain {_formatted(raw_means['final'])}. Raw equal-source delta: {_formatted(raw['source_weighted_delta_final_minus_stage1'])}. Raw numerical counts: chain lower {raw_counts['chain_better']}; PhantomSeal lower {raw_counts['ps_better']}; ties {raw_counts['ties']}.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Raw complete</th><th>Clean swap</th><th>PhantomSeal</th><th>Chain</th><th>Raw final − PS</th></tr></thead><tbody>{raw_condition_rows}</tbody></table></div></details></div>
<div class="panel"><h2>Detection and missing scores</h2><p>Output detection failures count only swaps attempted after a successful source detection.</p><div class="scroll"><table><tr><th>Method</th><th>Source failed</th><th>Output failed</th><th>Not attempted</th><th>Flags missing/invalid</th><th>Null/non-finite scores</th></tr>{failures_html}</table></div></div>
<div class="panel"><h2>All source / target pairs</h2><p>Every original score and preview remains visible. Weak baselines are inconclusive and have no primary effect direction; raw numerical differences remain in JSON/CSV. Preview images may be resized; scores come from the original benchmark files.</p><div class="scroll"><table><thead><tr><th>Source / target</th><th>Clean source</th><th>Fixed target</th><th>Clean swap</th><th>PhantomSeal swap</th><th>Chain swap</th><th>Qualified final − PS / inclusion</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></div>
<div class="panel"><h2>Representative previews</h2><p>Pairs selected at evenly spaced positions in sorted source/target ID order, independent of score and detection outcomes.</p>{contact_html}</div>
</main></html>'''
    html_path = output / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return {"summary": summary, "json_path": str(json_path), "csv_path": str(csv_path),
            "html_path": str(html_path), "contact_sheet_path": contact_result}
