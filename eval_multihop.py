#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)

    if isinstance(ground_truth, list):
        gold_answers = ground_truth
    else:
        gold_answers = [ground_truth]

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for gold in gold_answers:
        normalized_ground_truth = normalize_answer(gold)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    return best_f1, best_precision, best_recall


def em_and_acc(prediction: str, ground_truth) -> Tuple[int, int]:
    em = 0
    acc = 0
    normalized_prediction = normalize_answer(prediction)

    if isinstance(ground_truth, list):
        gold_answers = ground_truth
    else:
        gold_answers = [ground_truth]

    for gold in gold_answers:
        normalized_answer = normalize_answer(gold)
        if normalized_answer == normalized_prediction:
            em = 1
        if normalized_answer in normalized_prediction:
            acc = 1
        if em == 1 and acc == 1:
            break

    return em, acc


def calculate_em_f1_acc(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total_em = 0
    total_acc = 0
    total_f1 = 0
    count = 0

    for result in results:
        predicted = result.get('predicted_answer', '')
        gold = result.get('gold_answer', '')

        if predicted and gold:
            em, acc = em_and_acc(predicted, gold)
            f1, _, _ = f1_score(predicted, gold)

            total_em += em
            total_acc += acc
            total_f1 += f1
            count += 1

    if count == 0:
        return {
            'samples': 0,
            'em': 0,
            'acc': 0,
            'f1': 0,
        }

    return {
        'samples': count,
        'em': total_em / count,
        'acc': total_acc / count,
        'f1': total_f1 / count,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='evaluate multiihop results of EM and F1')
    parser.add_argument('--file', type=str, required=True, help='result file path (jsonl format)')

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: - {args.file}")
        exit(1)

    results = []
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    metrics = calculate_em_f1_acc(results)

    print(f"Total samples: {metrics['samples']}")
    print(f"EM (Exact Match): {metrics['em']:.4f}")
    print(f"ACC (Containment): {metrics['acc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")