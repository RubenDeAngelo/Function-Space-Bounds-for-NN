"""Validate one saved two-bump experiment or a complete paper suite."""
import argparse
import json
from pathlib import Path
from experiment_records import validate_record, dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args();folder=args.directory
    if (folder/'config.json').exists():
        from exp_verified_studies import validate_suite
        validate_suite(folder);return
    from exp_verified_two_bumps import load_saved_network
    from verified_arithmetic import Arithmetic
    manifest=json.loads((folder/'manifest.json').read_text())
    load_saved_network(folder,Arithmetic(manifest['precision_bits']))
    checks={policy:validate_record(json.loads((folder/f'{policy}.json').read_text()),manifest['domain']) for policy in ('adaptive','uniform')}
    dump(folder/'validation.json',checks);print(checks)


if __name__=='__main__':main()
