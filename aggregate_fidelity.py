"""Aggregate independent held-out galaxy jobs; preserve galaxy as sample unit."""
from pathlib import Path
import csv
import json
from scientific_fidelity_benchmark import GALAXIES, summarize, write_csv

FLOAT = ('flux_true','paired_aperture_response','paired_centroid_error_px',
         'blind_local_fit_flux_ratio','blind_control_flux_ratio','image_psnr_db')
INTEGER = ('psf_nr','seed','target_snr','site','fixed_site_detected_5mad',
           'null_test_false_positive_count','null_test_count')


def main():
    rows=[]
    root=Path('artifacts')
    actual=set()
    for name in GALAXIES:
        path=root/f'galaxy-{name}'/'measurements.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline='') as file:
            group=list(csv.DictReader(file))
        if len(group)!=2*2*2*4*4*12:
            raise ValueError(f'{name}: expected 1536 got {len(group)}')
        for entry in group:
            if entry['galaxy']!=name:
                raise ValueError('Incorrect test galaxy in source artifact')
            for k in FLOAT: entry[k]=float(entry[k])
            for k in INTEGER: entry[k]=int(entry[k])
        actual.add(name)
        rows.extend(group)
        print('VERIFIED_GALAXY',name,len(group),flush=True)
    if actual!=set(GALAXIES):
        raise AssertionError('Missing galaxies')
    # Must be 6 galaxies * 4 imaging regimes * 2 seeds * 4 SNR levels
    # * 4 methods * 12 sites = 9216 rows.
    if len(rows)!=9216:
        raise AssertionError(f'Unexpected row count {len(rows)}')
    summary=summarize(rows)
    write_csv('results/measurements.csv',rows)
    write_csv('results/summary.csv',summary)
    for item in summary:
        print('SUMMARY '+json.dumps(item,sort_keys=True),flush=True)
    print('EVALUATION_COMPLETE',len(rows),len(summary),flush=True)

if __name__=='__main__':
    main()