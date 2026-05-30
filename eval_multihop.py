#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""多跳问答评估模块 - 评估多跳推理结果的各种指标"""
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

MAX_HOPS = 5


def normalize_answer(s: str) -> str:
    """标准化答案文本，用于比较"""
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)  # 去掉冠词

    def white_space_fix(text):
        return " ".join(text.split())  # 去掉多余的空格

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)  # 去掉标点符号

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth) -> Tuple[float, float, float]:
    """计算F1分数、精确率和召回率
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案（可以是字符串或字符串列表）
    
    Returns:
        (f1, precision, recall): 取所有可能答案中的最高分
    """
    normalized_prediction = normalize_answer(prediction)
    
    # 处理 ground_truth 为列表的情况
    if isinstance(ground_truth, list):
        gold_answers = ground_truth
    else:
        gold_answers = [ground_truth]
    
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for gold in gold_answers:
        normalized_ground_truth = normalize_answer(gold)
        
        # if (
        #     normalized_prediction in ["yes", "no", "noanswer"]
        #     and normalized_prediction != normalized_ground_truth
        # ):
        #     continue
        # if (
        #     normalized_ground_truth in ["yes", "no", "noanswer"]
        #     and normalized_prediction != normalized_ground_truth
        # ):
        #     continue
        
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        # 取最高分
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    
    return best_f1, best_precision, best_recall


def em_and_acc(prediction: str, ground_truth) -> Tuple[int, int]:
    """计算EM（完全匹配）和ACC（包含匹配）
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案（可以是字符串或字符串列表）
    
    Returns:
        (em, acc): 只要匹配列表中任意一个答案即算正确
    """
    em = 0
    acc = 0
    normalized_prediction = normalize_answer(prediction)
    
    # 处理 ground_truth 为列表的情况
    if isinstance(ground_truth, list):
        gold_answers = ground_truth
    else:
        gold_answers = [ground_truth]
    
    for gold in gold_answers:
        normalized_answer = normalize_answer(gold)
        # if normalized_prediction in normalized_answer:
        if normalized_answer == normalized_prediction:
            em = 1
        if normalized_answer in normalized_prediction:
            acc = 1
        # 如果已经找到匹配，可以提前退出
        if em == 1 and acc == 1:
            break
    
    return em, acc


def calculate_em_f1_acc(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算EM、F1、ACC等基本指标"""
    total_em = 0
    total_acc = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    count = 0
    
    for result in results:
        predicted = result.get('predicted_answer', '')
        gold = result.get('gold_answer', '')
        
        if predicted and gold:
            em, acc = em_and_acc(predicted, gold)
            f1, precision, recall = f1_score(predicted, gold)
            
            total_em += em
            total_acc += acc
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            count += 1
    
    if count == 0:
        return {
            'samples': 0,
            'em': 0,
            'acc': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0
        }
    
    return {
        'samples': count,
        'em': total_em / count,
        'acc': total_acc / count,
        'f1': total_f1 / count,
        'precision': total_precision / count,
        'recall': total_recall / count
    }


def calculate_hop_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算不同跳数的统计数据"""
    # Extract original hop count from sample_id (e.g., "2hop__10017_18974" -> 2)
    hop_counts = defaultdict(int)
    hop_to_metrics = defaultdict(lambda: {'em': 0, 'acc': 0, 'f1': 0, 'count': 0})
    
    for result in results:
        sample_id = result.get('sample_id', '')
        # Extract the original hop count from sample_id (starts with "Xhop")
        import re
        match = re.match(r'(\d+)hop', sample_id)
        original_hop_count = int(match.group(1)) if match else None
        
        # Only count if we can extract the original hop count
        if original_hop_count is not None:
            hop_counts[original_hop_count] += 1
            
            predicted = result.get('predicted_answer', '')
            gold = result.get('gold_answer', '')
            
            if predicted and gold:
                em, acc = em_and_acc(predicted, gold)
                f1, _, _ = f1_score(predicted, gold)
                
                hop_to_metrics[original_hop_count]['em'] += em
                hop_to_metrics[original_hop_count]['acc'] += acc
                hop_to_metrics[original_hop_count]['f1'] += f1
                hop_to_metrics[original_hop_count]['count'] += 1
    
    # Calculate average metrics per original hop
    hop_stats = {}
    for hop_num, metrics in hop_to_metrics.items():
        count = metrics['count']
        if count > 0:
            hop_stats[f"{hop_num}_hop"] = {
                'samples': count,
                'em': metrics['em'] / count,
                'acc': metrics['acc'] / count,
                'f1': metrics['f1'] / count
            }
    
    # Also return overall distribution
    hop_distribution = dict(hop_counts)
    
    return {
        'hop_stats': hop_stats,
        'hop_distribution': hop_distribution
    }


def calculate_retrieval_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算检索相关的统计数据"""
    total_steps = 0
    retrieval_steps = 0  # 步骤中需要检索的次数
    internal_knowledge_steps = 0  # 步骤中使用内部知识的次数
    
    # Track per-sample statistics
    sample_retrieval_counts = []  # 每个样本的检索次数
    sample_internal_counts = []   # 每个样本使用内部知识的次数
    
    for result in results:
        hop_details = result.get('hop_details', [])
        sample_retrieval_count = 0
        sample_internal_count = 0
        
        for hop_detail in hop_details:
            # Count retrieval vs internal knowledge decisions
            context_judge = hop_detail.get('context_judge', False)
            if context_judge or hop_detail.get('hop_index', 'exceed') == 'exceed': # 直接回答了原问题，没有子问题;或者超出了跳数
                continue
            sub_q_judge = hop_detail.get('sub_q_judge', False)  # True means knows answer internally
            if sub_q_judge:
                internal_knowledge_steps += 1
                sample_internal_count += 1
            else:
                retrieval_steps += 1
                sample_retrieval_count += 1
        
        total_steps += len(hop_details)
        sample_retrieval_counts.append(sample_retrieval_count)
        sample_internal_counts.append(sample_internal_count)
    
    return {
        'total_steps': total_steps,
        'retrieval_steps': retrieval_steps,
        'internal_knowledge_steps': internal_knowledge_steps,
        'retrieval_ratio': retrieval_steps / total_steps if total_steps > 0 else 0,
        'internal_knowledge_ratio': internal_knowledge_steps / total_steps if total_steps > 0 else 0,
        'avg_retrieval_per_sample': sum(sample_retrieval_counts) / len(results),
        'avg_internal_per_sample': sum(sample_internal_counts) / len(results),
        'sample_retrieval_counts': sample_retrieval_counts,
        'sample_internal_counts': sample_internal_counts
    }


def calculate_time_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算时间相关的统计数据
    
    统计以下时间指标:
    - context_judge_time: context_judge 模型生成时间
    - sub_q_judge_time: sub_q_judge 模型生成时间
    - retrieval_time: 检索时间
    - compress_time: 压缩时间
    - sub_q_answer_time: sub_q_answer 模型生成时间
    - force_answer_time: force_answer 模型生成时间（在超出最大跳数时）
    """
    # 收集所有时间数据
    context_judge_times = []
    sub_q_judge_times = []
    retrieval_times = []
    compress_times = []
    sub_q_answer_times = []
    force_answer_times = []
    
    # 每个样本的总时间
    sample_total_times = []
    
    for result in results:
        hop_details = result.get('hop_details', [])
        sample_time = 0.0
        
        for hop_detail in hop_details:
            # context_judge_time: 每个 hop 都有
            if 'context_judge_time' in hop_detail:
                context_judge_times.append(hop_detail['context_judge_time'])
                sample_time += hop_detail['context_judge_time']
            
            # sub_q_judge_time: 当 context_judge 为 False 时（需要问子问题）
            if 'sub_q_judge_time' in hop_detail:
                sub_q_judge_times.append(hop_detail['sub_q_judge_time'])
                sample_time += hop_detail['sub_q_judge_time']
            
            # retrieval_time: 当 sub_q_judge 为 False 时（需要检索）
            if 'retrieval_time' in hop_detail:
                retrieval_times.append(hop_detail['retrieval_time'])
                sample_time += hop_detail['retrieval_time']
            
            # compress_time: 当使用 compressor 时
            if 'compress_time' in hop_detail:
                compress_times.append(hop_detail['compress_time'])
                sample_time += hop_detail['compress_time']
            
            # sub_q_answer_time: 当需要检索时
            if 'sub_q_answer_time' in hop_detail:
                sub_q_answer_times.append(hop_detail['sub_q_answer_time'])
                sample_time += hop_detail['sub_q_answer_time']
            
            # force_answer_time: 只在最后一个 hop（超出最大跳数时）
            if 'force_answer_time' in hop_detail:
                force_answer_times.append(hop_detail['force_answer_time'])
                sample_time += hop_detail['force_answer_time']
        
        sample_total_times.append(sample_time)
    
    def calc_stats(times: List[float]) -> Dict[str, float]:
        """计算时间列表的统计指标"""
        if not times:
            return {
                'count': 0,
                'total': 0.0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        sorted_times = sorted(times)
        n = len(sorted_times)
        median = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return {
            'count': n,
            'total': sum(times),
            'mean': sum(times) / n,
            'min': min(times),
            'max': max(times),
            'median': median
        }
    
    # 计算各类时间的统计指标
    context_judge_stats = calc_stats(context_judge_times)
    sub_q_judge_stats = calc_stats(sub_q_judge_times)
    retrieval_stats = calc_stats(retrieval_times)
    compress_stats = calc_stats(compress_times)
    sub_q_answer_stats = calc_stats(sub_q_answer_times)
    force_answer_stats = calc_stats(force_answer_times)
    sample_total_stats = calc_stats(sample_total_times)
    
    # 计算总体时间统计
    total_inference_time = (
        context_judge_stats['total'] +
        sub_q_judge_stats['total'] +
        force_answer_stats['total']
    )
    total_retrieval_time = retrieval_stats['total'] + sub_q_answer_stats['total']
    total_time = total_inference_time + total_retrieval_time
    
    return {
        'context_judge_time': context_judge_stats,
        'sub_q_judge_time': sub_q_judge_stats,
        'retrieval_time': retrieval_stats,
        'compress_time': compress_stats,
        'sub_q_answer_time': sub_q_answer_stats,
        'force_answer_time': force_answer_stats,
        'sample_total_time': sample_total_stats,
        'summary': {
            'total_inference_time': total_inference_time,
            'total_retrieval_time': total_retrieval_time,
            'total_compress_time': compress_stats['total'],
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(results) if results else 0,
            'inference_time_ratio': total_inference_time / total_time if total_time > 0 else 0,
            'retrieval_time_ratio': total_retrieval_time / total_time if total_time > 0 else 0
        }
    }


def calculate_compression_statistics(results: List[Dict[str, Any]], file_path: str = None) -> Dict[str, Any]:
    """计算检索压缩率统计
    
    从文件名中提取 topk 值，统计每次检索实际返回的文档数量，计算压缩率。
    压缩率 = 实际返回文档数 / topk值
    
    Args:
        results: 结果列表
        file_path: 结果文件路径，用于提取 topk 值
    
    Returns:
        压缩率统计字典
    """
    import os
    
    default_topk = 10
    
    if file_path:
        filename = os.path.basename(file_path)
        match = re.search(r'topk(\d+)', filename)
        if match:
            default_topk = int(match.group(1))
    
    total_retrieval_count = 0
    total_docs_retrieved = 0
    total_docs_expected = 0
    
    docs_per_retrieval = []
    compression_rates = []
    
    per_hop_stats = defaultdict(lambda: {'count': 0, 'docs_retrieved': 0, 'docs_expected': 0})
    
    for result in results:
        hop_details = result.get('hop_details', [])
        
        for hop_detail in hop_details:
            if 'retrieved_context' in hop_detail:
                retrieved_context = hop_detail['retrieved_context']
                actual_docs = len(retrieved_context) if retrieved_context else 0
                
                if actual_docs > 0:
                    total_retrieval_count += 1
                    total_docs_retrieved += actual_docs
                    total_docs_expected += default_topk
                    
                    docs_per_retrieval.append(actual_docs)
                    compression_rates.append(actual_docs / default_topk)
                    
                    hop_index = hop_detail.get('hop_index', -1)
                    per_hop_stats[hop_index]['count'] += 1
                    per_hop_stats[hop_index]['docs_retrieved'] += actual_docs
                    per_hop_stats[hop_index]['docs_expected'] += default_topk
    
    avg_docs_per_retrieval = total_docs_retrieved / total_retrieval_count if total_retrieval_count > 0 else 0
    avg_compression_rate = sum(compression_rates) / len(compression_rates) if compression_rates else 0
    
    per_hop_compression = {}
    for hop_idx, stats in sorted(per_hop_stats.items()):
        if stats['count'] > 0:
            per_hop_compression[f"hop_{hop_idx}"] = {
                'retrieval_count': stats['count'],
                'avg_docs': stats['docs_retrieved'] / stats['count'],
                'compression_rate': stats['docs_retrieved'] / stats['docs_expected']
            }
    
    docs_distribution = Counter(docs_per_retrieval)
    
    return {
        'configured_topk': default_topk,
        'total_retrieval_count': total_retrieval_count,
        'total_docs_retrieved': total_docs_retrieved,
        'total_docs_expected': total_docs_expected,
        'avg_docs_per_retrieval': avg_docs_per_retrieval,
        'overall_compression_rate': total_docs_retrieved / total_docs_expected if total_docs_expected > 0 else 0,
        'avg_compression_rate': avg_compression_rate,
        'docs_distribution': dict(sorted(docs_distribution.items())),
        'per_hop_compression': per_hop_compression
    }


def calculate_additional_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算其他有用的统计数据"""
    # Count how many samples reached max hops
    max_hops_samples = 0
    exceeded_hops_samples = 0  # Samples that exceeded max hops
    
    # Track hop efficiency (how well each hop contributes to the final answer)
    hop_efficiency_data = []
    
    for result in results:
        num_hops = result.get('num_hops', 0)
        # Assuming if num_hops >= max_hops (usually 8), it means it hit the limit
        if num_hops >= MAX_HOPS:
            max_hops_samples += 1
        if num_hops > MAX_HOPS: 
            exceeded_hops_samples += 1
        
        # Calculate hop efficiency
        predicted = result.get('predicted_answer', '')
        gold = result.get('gold_answer', '')
        if predicted and gold:
            em, _ = em_and_acc(predicted, gold)
            # Efficiency could be seen as successful hops vs total hops
            if num_hops > 0:
                hop_efficiency_data.append(em / num_hops)  # Higher is better if answer is correct
    
    # Calculate answer quality indicators
    exact_match_count = 0
    partial_match_count = 0
    no_answer_count = 0  # Cases where predicted answer is empty or indicates failure
    
    for result in results:
        predicted = result.get('predicted_answer', '')
        gold = result.get('gold_answer', '')
        
        if predicted and gold:
            em, acc = em_and_acc(predicted, gold)
            if em == 1:
                exact_match_count += 1
            elif acc == 1:
                partial_match_count += 1
        else:
            no_answer_count += 1
    
    # Calculate hop efficiency metrics
    avg_hop_efficiency = sum(hop_efficiency_data) / len(hop_efficiency_data) if hop_efficiency_data else 0
    
    return {
        'total_samples': len(results),
        'samples_hit_max_hops': max_hops_samples,
        'samples_exceeded_hops': exceeded_hops_samples,
        'max_hops_ratio': max_hops_samples / len(results) if results else 0,
        'exact_matches': exact_match_count,
        'partial_matches': partial_match_count,
        'no_answers': no_answer_count,
        'exact_match_ratio': exact_match_count / len(results) if results else 0,
        'partial_match_ratio': partial_match_count / len(results) if results else 0,
        'no_answer_ratio': no_answer_count / len(results) if results else 0,
        'avg_hop_efficiency': avg_hop_efficiency
    }


def evaluate_multihop_results(file_path: str) -> Dict[str, Any]:
    """评估多跳推理结果的主函数，返回所有指标
    
    Args:
        file_path: 结果文件路径（jsonl格式）
    
    Returns:
        包含所有评估指标的字典
    """
    # Load results
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Calculate all metrics
    basic_metrics = calculate_em_f1_acc(results)
    hop_stats = calculate_hop_statistics(results)
    retrieval_stats = calculate_retrieval_statistics(results)
    time_stats = calculate_time_statistics(results)
    additional_stats = calculate_additional_statistics(results)
    compression_stats = calculate_compression_statistics(results, file_path)
    
    # Combine all metrics
    evaluation_results = {
        'basic_metrics': basic_metrics,
        'hop_statistics': hop_stats,
        'retrieval_statistics': retrieval_stats,
        'time_statistics': time_stats,
        'additional_statistics': additional_stats,
        'compression_statistics': compression_stats,
        'summary': {
            'total_samples': len(results),
            'avg_hops_used': sum(r.get('num_hops', 0) for r in results) / len(results) if results else 0,
            'em': basic_metrics['em'],
            'f1': basic_metrics['f1'],
            'acc': basic_metrics['acc'],
            'avg_time_per_sample': time_stats['summary']['avg_time_per_sample'],
            'total_time': time_stats['summary']['total_time'],
            'overall_compression_rate': compression_stats['overall_compression_rate']
        }
    }
    
    return evaluation_results


def print_evaluation_report(evaluation_results: Dict[str, Any], results: List[Dict[str, Any]] = None):
    """打印评估报告"""
    print("=" * 80)
    print("多跳推理评估报告")
    print("=" * 80)
    
    # Basic metrics
    basic = evaluation_results['basic_metrics']
    print(f"\n基础指标:")
    print(f"  样本数: {basic['samples']}")
    print(f"  EM (Exact Match): {basic['em']:.4f}")
    print(f"  ACC (Answer Containment): {basic['acc']:.4f}")
    print(f"  F1 Score: {basic['f1']:.4f}")
    print(f"  Precision: {basic['precision']:.4f}")
    print(f"  Recall: {basic['recall']:.4f}")
    
    # Hop statistics
    hop_stats = evaluation_results['hop_statistics']
    print(f"\n原始问题跳数统计:")
    print(f"  平均实际跳数: {evaluation_results['summary']['avg_hops_used']:.2f}")
    print("  各原始跳数性能:")
    for hop_key, metrics in sorted(hop_stats['hop_stats'].items(), key=lambda x: int(x[0].split('_')[0])):
        print(f"    {hop_key}: EM={metrics['em']:.4f}, ACC={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, samples={metrics['samples']}")
    
    # Also show actual vs expected hop statistics if results are provided
    if results is not None:
        print(f"\n实际运行跳数统计:")
        actual_hop_counts = defaultdict(int)
        for result in results:
            actual_hop_counts[result.get('num_hops', 0)] += 1
        actual_hop_counts = dict(sorted(actual_hop_counts.items()))
        print(f"  实际运行跳数分布: {actual_hop_counts}")
    
    # Retrieval statistics
    retrieval = evaluation_results['retrieval_statistics']
    print(f"\n检索统计:")
    print(f"  总步骤数: {retrieval['total_steps']}")
    print(f"  检索步骤数: {retrieval['retrieval_steps']} ({retrieval['retrieval_ratio']:.2%})")
    print(f"  内部知识步骤数: {retrieval['internal_knowledge_steps']} ({retrieval['internal_knowledge_ratio']:.2%})")
    print(f"  平均每样本检索次数: {retrieval['avg_retrieval_per_sample']:.2f}")
    print(f"  平均每样本内部知识使用次数: {retrieval['avg_internal_per_sample']:.2f}")
    
    # Compression statistics
    if 'compression_statistics' in evaluation_results:
        compression = evaluation_results['compression_statistics']
        print(f"\n压缩率统计:")
        print(f"  配置的 topk 值: {compression['configured_topk']}")
        print(f"  总检索次数: {compression['total_retrieval_count']}")
        print(f"  平均每次检索文档数: {compression['avg_docs_per_retrieval']:.2f}")
        print(f"  总体压缩率: {compression['overall_compression_rate']:.2%}")
        print(f"  文档数量分布: {compression['docs_distribution']}")
        if compression['per_hop_compression']:
            print(f"  各跳压缩率:")
            for hop_key, hop_stats in sorted(compression['per_hop_compression'].items()):
                print(f"    {hop_key}: 平均文档数={hop_stats['avg_docs']:.2f}, 压缩率={hop_stats['compression_rate']:.2%}, 检索次数={hop_stats['retrieval_count']}")
    
    # Time statistics
    time_stats = evaluation_results['time_statistics']
    print(f"\n时间统计:")
    print(f"  总推理时间: {time_stats['summary']['total_inference_time']:.2f}s")
    print(f"  总检索时间: {time_stats['summary']['total_retrieval_time']:.2f}s")
    print(f"  总压缩时间: {time_stats['summary']['total_compress_time']:.2f}s")
    print(f"  总时间: {time_stats['summary']['total_time']:.2f}s")
    print(f"  平均每样本时间: {time_stats['summary']['avg_time_per_sample']:.2f}s")
    print(f"  推理时间占比: {time_stats['summary']['inference_time_ratio']:.2%}")
    print(f"  检索时间占比: {time_stats['summary']['retrieval_time_ratio']:.2%}")
    
    # Detailed time statistics
    print(f"\n详细时间统计 (均值 ± 标准差 [min - max]):")
    for time_name, time_key in [
        ('Context Judge', 'context_judge_time'),
        ('Sub-Q Judge', 'sub_q_judge_time'),
        ('Retrieval', 'retrieval_time'),
        ('Compress', 'compress_time'),
        ('Sub-Q Answer', 'sub_q_answer_time'),
        ('Force Answer', 'force_answer_time')
    ]:
        stats = time_stats[time_key]
        if stats['count'] > 0:
            print(f"  {time_name}: {stats['mean']:.3f}s [{stats['min']:.3f}s - {stats['max']:.3f}s], count={stats['count']}")
    
    # Additional statistics
    additional = evaluation_results['additional_statistics']
    print(f"\n其他统计:")
    print(f"  达到最大跳数的样本: {additional['samples_hit_max_hops']} ({additional['max_hops_ratio']:.2%})")
    print(f"  完全匹配数: {additional['exact_matches']} ({additional['exact_match_ratio']:.2%})")
    print(f"  部分匹配数: {additional['partial_matches']} ({additional['partial_match_ratio']:.2%})")
    print(f"  无答案数: {additional['no_answers']} ({additional['no_answer_ratio']:.2%})")
    print(f"  平均跳效率: {additional['avg_hop_efficiency']:.4f}")


def save_evaluation_results(evaluation_results: Dict[str, Any], output_path: str):
    """保存评估结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {output_path}")


def compare_multiple_files(file_paths: List[str]) -> Dict[str, Any]:
    """比较多个结果文件的指标
    
    Args:
        file_paths: 多个结果文件路径列表
    
    Returns:
        比较结果的字典
    """
    comparison_results = {}
    
    for file_path in file_paths:
        file_name = Path(file_path).stem
        print(f"正在评估文件: {file_path}")
        evaluation_result = evaluate_multihop_results(file_path)
        comparison_results[file_name] = {
            'file_path': file_path,
            'metrics': evaluation_result['summary'],
            'evaluation_results': evaluation_result  # Store full results for comparison
        }
    
    return comparison_results


def print_comparison_report(comparison_results: Dict[str, Any]):
    """打印比较报告"""
    print("=" * 100)
    print("多跳推理结果比较报告")
    print("=" * 100)
    
    print(f"{'模型':<30} {'样本数':<8} {'EM':<8} {'F1':<8} {'ACC':<8} {'平均跳数':<10} {'检索比例':<10} {'成功率':<10}")
    print("-" * 100)
    
    for model_name, result in comparison_results.items():
        metrics = result['metrics']
        retrieval_stats = result['evaluation_results']['retrieval_statistics']
        additional_stats = result['evaluation_results']['additional_statistics']
        print(f"{model_name:<30} {metrics['total_samples']:<8} {metrics['em']:<8.4f} {metrics['f1']:<8.4f} "
              f"{metrics['acc']:<8.4f} {metrics['avg_hops_used']:<10.2f} {retrieval_stats['retrieval_ratio']:<10.2%} "
              f"{additional_stats['sub_question_success_rate']:<10.2%}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估多跳推理结果')
    parser.add_argument('--file', type=str, default='empirical/results_multihop/2wiki/Qwen_Qwen2.5-32B-Instruct/multihop7_hops5_topk80_rerank0.2_bs1_entropy01.jsonl', help='结果文件路径（jsonl格式）')
    parser.add_argument('--compare', nargs='+', help='多个结果文件路径，用于比较')
    parser.add_argument('--output', type=str, default='empirical/results_multihop/2wiki/Qwen_Qwen2.5-32B-Instruct/580_rerank02_bs1_entropy01.jsonl', help='输出结果文件路径（可选，支持.json格式）')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple files
        comparison_results = compare_multiple_files(args.compare)
        print_comparison_report(comparison_results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            print(f"\n比较结果已保存至: {args.output}")
    else:
        # Evaluate single file
        if not args.file:
            print("错误：必须提供 --file 或 --compare 参数")
            exit(1)
            
        print(f"正在评估文件: {args.file}")
        # Load the original results to pass to print function
        original_results = []
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    original_results.append(json.loads(line))
        
        evaluation_results = evaluate_multihop_results(args.file)
        print_evaluation_report(evaluation_results, original_results)
        
        if args.output:
            save_evaluation_results(evaluation_results, args.output)