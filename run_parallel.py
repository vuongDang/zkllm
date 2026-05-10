import os
import sys
import json
import argparse
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

SYMLINKS = [
    'rmsnorm', 'self-attn', 'ffn', 'skip-connection',
    'zkllm-workdir', 'swiglu-table.bin', 'fileio_utils.py',
]


def run_one(args):
    idx, prompt, run_dir, max_new_tokens = args
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / 'prompt.json').write_text(json.dumps([prompt]))

    for name in SYMLINKS:
        src = BASE_DIR / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    log_path = run_dir / 'run.log'
    with open(log_path, 'w') as log:
        ret = subprocess.run(
            ['conda', 'run', '-n', 'zkllm-env', 'python',
             str(BASE_DIR / 'full_run.py'), 'prompt.json',
             '--max_new_tokens', str(max_new_tokens), '--skip-build'],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if ret.returncode != 0:
        print(f'[prompt {idx}] FAILED — see {log_path}')
        return idx, None

    result = json.loads((run_dir / 'output.json').read_text())
    print(f'[prompt {idx}] done')
    return idx, result[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full_run.py in parallel across prompts')
    parser.add_argument('input_file')
    parser.add_argument('--max_new_tokens', type=int, default=1)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    if os.system('make all') != 0:
        print('Build failed'); sys.exit(1)

    prompts = json.load(open(args.input_file))
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        print('Error: JSON file must contain a list of strings'); sys.exit(1)

    run_base = BASE_DIR / '_parallel_runs'
    tasks = [
        (i, p, str(run_base / f'run_{i}'), args.max_new_tokens)
        for i, p in enumerate(prompts)
    ]

    print(f'Running {len(prompts)} prompt(s) with {args.workers} workers...')
    wall_start = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_one, tasks)
    wall_time = time.time() - wall_start

    results.sort(key=lambda x: x[0])
    output = [r for _, r in results if r is not None]

    with open('parallel_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results written to output.json ({len(output)}/{len(prompts)} succeeded)')

    if output:
        n          = len(output)
        matches    = sum(1 for r in output if r['match'])
        total_zk   = sum(r['zk_time'] for r in output)
        total_base = sum(r['base_time'] for r in output)
        print(f'\n--- Summary ({n} prompt(s)) ---')
        print(f'Wall time:         {wall_time:.1f}s')
        print(f'ZK inference:      {total_zk:.1f}s total, {total_zk/n:.1f}s avg')
        print(f'Vanilla inference: {total_base:.1f}s total, {total_base/n:.1f}s avg')
        print(f'Output match:      {matches}/{n} ({100*matches/n:.0f}%)')
