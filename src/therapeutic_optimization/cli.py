#I do not understand the parsing here!!!
#TODO: determine what the parsing here is doing, replicate to fit my configuration


from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    ComplexSearchConfig,
    DEFAULT_ESM2_MODEL,
    ESM2AnalysisConfig,
    MutationConfig,
    PredictorConfig,
    WorkflowConfig,
)
from .io import normalize_sequence, read_single_fasta
from .workflow import OptimizationWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description='Run therapeutic optimization workflow stages.')
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--mode', choices=['basic', 'sophisticated'], default='basic')
    parser.add_argument('--target-survivors', type=int, default=25)
    parser.add_argument('--complex-replacement-order', nargs='+', default=['R', 'H', 'Q', 'E', 'C'])
    parser.add_argument('--min-mean-residue-cosine', type=float, default=0.9995)
    parser.add_argument('--max-perplexity-percent-change', type=float)
    parser.add_argument('--max-complex-candidates', type=int)
    parser.add_argument(
        '--stage',
        choices=['all', 'lysine-free', 'T1', 'UP1', 'T2', 'ESM2', 'S1', 'R1', 'UB2', 'R2'],
        default='all',
    )
    parser.add_argument('--sequence', type=str)
    parser.add_argument('--sequence-file', type=Path)
    parser.add_argument('--protein-id', default='WT')
    parser.add_argument('--threshold', type=float, default=0.40)
    parser.add_argument('--mutation-mode', choices=['single', 'combinatorial'], default='single')
    parser.add_argument('--replacement-aa', action='append', dest='replacement_aas')
    parser.add_argument('--max-combination-order', type=int, default=2)
    parser.add_argument('--esm2-model', default=DEFAULT_ESM2_MODEL)
    parser.add_argument('--esm2-mask-batch-size', type=int, default=4)
    parser.add_argument('--esm2-device')
    parser.add_argument(
        '--esm2-dtype',
        choices=['auto', 'float32', 'float16', 'bfloat16'],
        default='auto',
    )
    parser.add_argument('--esm2-perplexity-weight', type=float, default=0.70)
    parser.add_argument('--esm2-representation-weight', type=float, default=0.30)
    parser.add_argument('--no-esm2-per-residue', action='store_true')
    parser.add_argument(
        '--skip-esm2',
        action='store_true',
        help='For all: skip ESM-2 analysis and preserve the legacy R2 ranking.',
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='For S1/all/lysine-free: use existing structures instead of calling ColabFold.',
    )
    args = parser.parse_args()
    if args.mode == 'sophisticated' and args.skip_esm2:
        parser.error('--skip-esm2 is available only in basic mode.')

    replacements = tuple(args.replacement_aas or ['A'])
    config = WorkflowConfig(
        mode=args.mode,
        complex_search=ComplexSearchConfig(
            replacement_aas=tuple(args.complex_replacement_order),
            target_survivors=args.target_survivors,
            min_mean_residue_cosine_similarity=args.min_mean_residue_cosine,
            max_pseudo_perplexity_percent_change=args.max_perplexity_percent_change,
            max_candidates=args.max_complex_candidates,
        ),
        mutation=MutationConfig(
            threshold=args.threshold,
            mode=args.mutation_mode,
            replacement_aas=replacements,
            max_combination_order=args.max_combination_order,
        ),
        ubiquitination=PredictorConfig(threshold=args.threshold),
        esm2=ESM2AnalysisConfig(
            model_name=args.esm2_model,
            mask_batch_size=args.esm2_mask_batch_size,
            device=args.esm2_device,
            dtype=args.esm2_dtype,
            save_per_residue=not args.no_esm2_per_residue,
            perplexity_weight=args.esm2_perplexity_weight,
            representation_weight=args.esm2_representation_weight,
        ),
    )
    workflow = OptimizationWorkflow(args.project_root, config)

    sequence: str | None = None
    if args.sequence_file:
        raw = args.sequence_file.read_text(encoding='utf-8')
        if raw.lstrip().startswith('>'):
            _header, sequence = read_single_fasta(args.sequence_file)
        else:
            sequence = normalize_sequence(raw)
    elif args.sequence:
        sequence = normalize_sequence(args.sequence)

    if args.stage == 'lysine-free':
        if sequence is None and not workflow.paths.wt_fasta.exists():
            parser.error(
                '--sequence or --sequence-file is required for lysine-free unless '
                'storage/inputs/wt_input.fasta already exists.'
            )
        results = workflow.run_lysine_free_comparison(
            sequence=sequence,
            protein_id=args.protein_id,
            predict_structures=not args.analyze_only,
        )
        print(results['lysine_free_comparison'].to_string(index=False))
        print(
            'Comparison written to:',
            workflow.paths.table('LF1_WT_vs_all_K_to_R_comparison.csv'),
        )
        return

    if args.stage in {'T1', 'all'}:
        if sequence is None:
            parser.error('--sequence or --sequence-file is required for T1/all.')
        if args.stage == 'T1':
            workflow.T1(sequence, args.protein_id)
            return

    if args.stage == 'UP1':
        workflow.UP1()
    elif args.stage == 'T2':
        workflow.T2(predict_structures=not args.analyze_only)
    elif args.stage == 'ESM2':
        workflow.ESM2()
    elif args.stage == 'S1':
        workflow.S1(predict_structures=not args.analyze_only)
    elif args.stage == 'R1':
        workflow.R1()
    elif args.stage == 'UB2':
        workflow.UB2()
    elif args.stage == 'R2':
        workflow.R2()
    elif args.stage == 'all':
        workflow.run_all(
            sequence, protein_id=args.protein_id,
            predict_structures=not args.analyze_only,
            analyze_esm2=not args.skip_esm2,
        )


if __name__ == '__main__':
    main()
