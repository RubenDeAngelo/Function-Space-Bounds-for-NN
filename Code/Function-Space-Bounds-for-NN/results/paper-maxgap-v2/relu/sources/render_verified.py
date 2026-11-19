"""Render saved records only: no training and no interval range evaluation."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args();folder=args.directory
    if (folder/'config.json').exists():
        from exp_verified_studies import render_suite
        render_suite(folder);return
    from exp_verified_two_bumps import render
    manifest=json.loads((folder/'manifest.json').read_text())
    records={k:json.loads((folder/f'{k}.json').read_text()) for k in ('adaptive','uniform')}
    render(folder,records,manifest.get('norm','H2'),bool(manifest.get('reference_slope')))


if __name__=='__main__':main()
