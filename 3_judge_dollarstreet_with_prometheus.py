import os
import re
import glob
import json
import argparse
from tqdm import tqdm
from itertools import product
from collections import defaultdict

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

from benchmarks.dollarstreet import DollarStreet, VALID_COUNTRIES


def get_output_key(args):
    key_elems = [
        args.model.replace('/', '-').replace('_', '-'),
        os.path.basename(args.prompt_file).replace('.json', ''),
        args.judgment_lang
    ]

    if args.temperature != 0.0:
        key_elems.append('temp=%s' % args.temperature)
        key_elems.append('top-p=%s' % args.top_p)

    if args.self_consistency_reps != 1:
        assert args.temperature != 0.0
        key_elems.append('reps=%s' % args.self_consistency_reps)

    return '_'.join(key_elems)


def prepare_inputs(datasets, i, stratum, results, generation_lang, prompts, args):
    instruction = prompts['instruction'][args.judgment_lang]

    generation = results['test_gens'][stratum][i]

    label = datasets[(generation_lang, stratum)].examples[i]['label']
    assert label == results['candidates'][results['test_labels'][stratum][i]]

    # here we strip out non-English / non-Chinese characters which
    # affect the ability of the judgment model to score generations correctly
    if generation_lang == 'en':
        response = generation.replace('Љ', '').strip()
    else:
        response = re.sub(
            r'^[^\u4e00-\u9fff]+|[^\u4e00-\u9fff]+$',
            '', generation
        ).strip()

    return instruction, response, label


def load_to_process(datasets, prompts, output_key, args):
    to_process = {
        'idxs': [],
        'instructions': [],
        'responses': [],
        'reference_answers': []
    }
    num_files = 0
    for result_file in glob.glob(f'{args.output_dir}/dollar*{args.filter}/*results.json'):
        with open(result_file, 'r') as f:
            results = json.load(f)

        generation_lang = result_file.split('/')[-1].split('_')[0]
        if args.generation_lang.strip() != '' and generation_lang != args.generation_lang:
            continue

        if output_key in results and not args.force:
            continue

        with open(result_file, 'r') as f:
            results = json.load(f)

        for stratum in results['test_gens'].keys():
            assert len(datasets[(generation_lang, stratum)]) == len(results['test_gens'][stratum])

            if len(args.stratum_filter) > 0 and stratum not in args.stratum_filter:
                continue

            for i in range(len(results['test_gens'][stratum])):
                to_process['idxs'].append(
                    (result_file, stratum, i)
                )

                instruction, response, reference_answer = prepare_inputs(
                    datasets, i, stratum, results, generation_lang, prompts, args
                )

                to_process['instructions'].append(instruction)
                to_process['responses'].append(response)
                to_process['reference_answers'].append(reference_answer)

        num_files += 1
        if args.max_files and num_files == args.max_files:
            break

    print("found %s to process from %s files..." % (len(to_process['idxs']), num_files))

    return to_process


def batch_process(
    judge, instructions, responses, reference_answers,
    rubric, temperature, top_p, self_consistency_reps
):
    all_feedbacks, all_scores = [], []
    for seed in tqdm(range(self_consistency_reps), total=self_consistency_reps, desc='self-consistency'):
        feedbacks, scores = judge.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=rubric,
            reference_answers=reference_answers,
            params={
                "max_tokens": 1024,
                "repetition_penalty": 1.03,
                "best_of": 1,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed
            }
        )
        all_feedbacks.append(feedbacks)
        all_scores.append(scores)
    return all_feedbacks, all_scores


def write_scores(to_process, all_feedbacks, all_scores, output_key):
    print("writing scores...")

    assert len(to_process['idxs']) == len(all_feedbacks[0])
    assert len(to_process['idxs']) == len(all_scores[0])

    by_result_file_stratum_idxs = defaultdict(
        lambda: defaultdict(dict)
    )
    for feedback_score_idx, (result_file, stratum, idx) in enumerate(to_process['idxs']):
        by_result_file_stratum_idxs[result_file][stratum][idx] = (
            [
                feedbacks[feedback_score_idx] for feedbacks in all_feedbacks
            ],
            [
                scores[feedback_score_idx] for scores in all_scores
            ]
        )

    for result_file in by_result_file_stratum_idxs.keys():
        with open(result_file, 'r') as f:
            results = json.load(f)

        by_stratum = {}
        for stratum in results['test_gens'].keys():
            if stratum not in by_stratum:
                by_stratum[stratum] = []

            for idx in range(len(results['test_gens'][stratum])):
                feedbacks, scores = by_result_file_stratum_idxs[result_file][stratum][idx]
                by_stratum[stratum].append(
                    (feedbacks, scores)
                )

        results[output_key] = by_stratum

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)


def main(args):
    output_key = get_output_key(args)

    print("output_key: ", output_key)

    with open(args.prompt_file, 'r') as f:
        prompts = json.load(f)

    countries = list(sorted(VALID_COUNTRIES))
    datasets = {
        (lang, stratum): DollarStreet(
            ann_countries=[stratum], target_langs=[lang], label_set='dollar',
            splits=['test', 'train'], preprocess=lambda x: x, corpus_dir=args.resources_dir
        ) for lang, stratum in product(['en', 'zh'], countries)
    }

    to_process = load_to_process(datasets, prompts, output_key, args)

    if args.model == '8x7b':
        model = VLLM(model="prometheus-eval/prometheus-8x7b-v2.0", tensor_parallel_size=4)
        judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    else:
        raise NotImplementedError

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(
        **prompts['rubric'][args.judgment_lang]
    )

    # process outstanding result files

    all_feedbacks, all_scores = batch_process(
        judge,
        to_process['instructions'],
        to_process['responses'],
        to_process['reference_answers'],
        score_rubric, args.temperature,
        args.top_p, args.self_consistency_reps
    )
    write_scores(to_process, all_feedbacks, all_scores, output_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Judge DollarStreet generations with Prometheus2."
    )
    parser.add_argument('--model', type=str, required=True, choices=[
        '7b', '8x7b'
    ])
    parser.add_argument('--prompt-file', type=str, required=True)
    parser.add_argument('--generation-lang', type=str, default='')
    parser.add_argument('--judgment-lang', type=str, default='en')
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--self-consistency-reps', type=int, default=1)
    parser.add_argument('--stratum-filter', nargs='+', default=[])
    parser.add_argument('--max-files', type=int, default=-1)
    parser.add_argument('--resources-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    main(args)
