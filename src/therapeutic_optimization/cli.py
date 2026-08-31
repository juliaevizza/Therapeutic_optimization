from __future__ import annotations

import argparse
from pathlib import Path

from .config import MutationConfig, PredictorConfig, WorkflowConfig
from .io import normalize_sequence, read_single_fasta
from .workflow import OptimizationWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description='Run therapeutic optimization workflow stages.')
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument(
        '--stage',
        choices=['all', 'lysine-free', 'T1', 'UP1', 'T2', 'S1', 'R1', 'UB2', 'R2'],
        default='all',
    )
    parser.add_argument('--sequence', type=str)
    parser.add_argument('--sequence-file', type=Path)
    parser.add_argument('--protein-id', default='WT')
    parser.add_argument('--threshold', type=float, default=0.40)
    parser.add_argument('--mutation-mode', choices=['single', 'combinatorial'], default='single')
    parser.add_argument('--replacement-aa', action='append', dest='replacement_aas')
    parser.add_argument('--max-combination-order', type=int, default=2)
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='For S1/all/lysine-free: use existing structures instead of calling ColabFold.',
    )
    args = parser.parse_args()

    replacements = tuple(args.replacement_aas or ['A'])
    config = WorkflowConfig(
        mutation=MutationConfig(
            threshold=args.threshold,
            mode=args.mutation_mode,
            replacement_aas=replacements,
            max_combination_order=args.max_combination_order,
        ),
        ubiquitination=PredictorConfig(threshold=args.threshold),
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
            workflow.paths.tables / 'LF1_WT_vs_all_K_to_R_comparison.csv',
        )
        return

    if args.stage in {'T1', 'all'}:
        if sequence is None:
            parser.error('--sequence or --sequence-file is required for T1/all.')
        workflow.T1(sequence, args.protein_id)
        if args.stage == 'T1':
            return

    if args.stage == 'UP1':
        workflow.UP1()
    elif args.stage == 'T2':
        workflow.T2()
    elif args.stage == 'S1':
        workflow.S1(predict_structures=not args.analyze_only)
    elif args.stage == 'R1':
        workflow.R1()
    elif args.stage == 'UB2':
        workflow.UB2()
    elif args.stage == 'R2':
        workflow.R2()
    elif args.stage == 'all':
        up1 = workflow.UP1()
        t2 = workflow.T2(up1)
        s1_metrics, s1_conserved = workflow.S1(t2, predict_structures=not args.analyze_only)
        workflow.R1(t2, s1_metrics)
        ub2 = workflow.UB2(s1_conserved)
        workflow.R2(up1, ub2, s1_conserved)


if __name__ == '__main__':
    main()
