from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F


def build_powerset_mapping(num_classes: int, max_set_size: int) -> torch.Tensor:
    rows = []
    for set_size in range(0, max_set_size + 1):
        for current_set in combinations(range(num_classes), set_size):
            row = torch.zeros(num_classes, dtype=torch.float32)
            if current_set:
                row[list(current_set)] = 1.0
            rows.append(row)
    return torch.stack(rows, dim=0)


def powerset_logits_to_speech(batch_logits: torch.Tensor, mapping: torch.Tensor) -> np.ndarray:
    predicted = torch.argmax(batch_logits, dim=-1)
    one_hot = F.one_hot(predicted, num_classes=mapping.shape[0]).float()
    multilabel = torch.matmul(one_hot, mapping)
    speech = torch.max(multilabel, dim=-1, keepdim=True).values
    return speech.detach().cpu().numpy()


def _closest_frame(time: float, frame_start: float, frame_duration: float, frame_step: float) -> int:
    return int(np.rint((time - frame_start - 0.5 * frame_duration) / frame_step))


def aggregate_sliding_scores(
    scores: np.ndarray,
    chunk_duration: float,
    chunk_step: float,
    frame_start: float,
    frame_duration: float,
    frame_step: float,
    total_duration: float,
    warm_up: tuple[float, float] = (0.0, 0.0),
    hamming: bool = True,
    epsilon: float = 1e-12,
) -> np.ndarray:
    if scores.ndim != 3:
        raise ValueError("Expected scores with shape (num_chunks, num_frames_per_chunk, num_classes).")

    num_chunks, num_frames_per_chunk, num_classes = scores.shape

    hamming_window = np.hamming(num_frames_per_chunk).reshape(-1, 1) if hamming else np.ones((num_frames_per_chunk, 1))

    warm_up_window = np.ones((num_frames_per_chunk, 1))
    warm_up_left = round(warm_up[0] / chunk_duration * num_frames_per_chunk)
    warm_up_right = round(warm_up[1] / chunk_duration * num_frames_per_chunk)
    if warm_up_left > 0:
        warm_up_window[:warm_up_left] = epsilon
    if warm_up_right > 0:
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

    num_output_frames = max(1, int(np.floor(total_duration / frame_step)) + 1)
    aggregated_output = np.zeros((num_output_frames, num_classes), dtype=np.float32)
    overlapping_chunk_count = np.zeros((num_output_frames, num_classes), dtype=np.float32)
    aggregated_mask = np.zeros((num_output_frames, num_classes), dtype=np.float32)

    for chunk_index, chunk_scores in enumerate(scores):
        mask = 1 - np.isnan(chunk_scores)
        safe_scores = np.nan_to_num(chunk_scores, copy=True, nan=0.0)
        chunk_start = chunk_index * chunk_step
        start_frame = _closest_frame(chunk_start + 0.5 * frame_duration, frame_start, frame_duration, frame_step)
        end_frame = min(num_output_frames, start_frame + num_frames_per_chunk)
        local_slice = slice(0, end_frame - start_frame)
        target_slice = slice(start_frame, end_frame)
        weighted = mask[local_slice] * hamming_window[local_slice] * warm_up_window[local_slice]
        aggregated_output[target_slice] += safe_scores[local_slice] * weighted
        overlapping_chunk_count[target_slice] += weighted
        aggregated_mask[target_slice] = np.maximum(aggregated_mask[target_slice], mask[local_slice])

    average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)
    average[aggregated_mask == 0.0] = 0.0
    return average


def _merge_close_segments(segments: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def scores_to_segments(
    scores: np.ndarray,
    frame_start: float,
    frame_duration: float,
    frame_step: float,
    onset: float = 0.5,
    offset: float = 0.5,
    min_duration_on: float = 0.25,
    min_duration_off: float = 0.1,
) -> list[tuple[float, float]]:
    if scores.ndim != 1:
        raise ValueError("Expected a 1-D speech score array.")
    if len(scores) == 0:
        return []

    timestamps = frame_start + 0.5 * frame_duration + np.arange(len(scores)) * frame_step
    segments: list[tuple[float, float]] = []
    start = timestamps[0]
    is_active = scores[0] > onset
    last_timestamp = timestamps[0]

    for timestamp, score in zip(timestamps[1:], scores[1:]):
        last_timestamp = timestamp
        if is_active:
            if score < offset:
                segments.append((start, timestamp))
                start = timestamp
                is_active = False
        else:
            if score > onset:
                start = timestamp
                is_active = True

    if is_active:
        segments.append((start, last_timestamp))

    segments = _merge_close_segments(segments, min_duration_off)
    return [(start, end) for start, end in segments if end - start >= min_duration_on]
