"""Windows-friendly Neuro-Flow menu; independent seated/standing experiments."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import nf_core as nf
from segmented_scoring import scoring_dir

PROJECT_DIR = Path(__file__).resolve().parent


def run(script, *args):
    script_path = PROJECT_DIR / script
    print('\n>>> python', script_path, *map(str,args), '\n')
    try:
        subprocess.run([sys.executable, str(script_path), *map(str,args)], cwd=PROJECT_DIR, check=False)
    except KeyboardInterrupt:
        print('Interrupted')
    input('\nPress ENTER to return to the menu...')


def next_name(settings, category, prefix):
    folder = nf.mode_dir('clips', settings) / category
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while (folder / f'{prefix}_{n:02d}.avi').exists():
        n += 1
    return f'{prefix}_{n:02d}'


def main():
    while True:
        s = nf.load_settings()
        position = s['position']
        ref = s['reference_clip'] or 'NOT RECORDED'
        threshold = s['dtw_threshold'] if s['dtw_threshold'] is not None else 'NOT CALIBRATED'
        print('\n' + '='*68)
        print(f'NEURO-FLOW | {position.upper()} | shoulder abduction')
        print(f'Arm: {s["arm"]} | Target: {s["movement_target_deg"]} deg | DTW: {threshold}')
        print(f'Reference: {ref}')
        print(f'Recording: {s["record_countdown_s"]}s countdown | automatic stop | {s["rep_timeout_s"]}s movement timeout')
        print('='*68)
        print(' 1  Live skeleton (S screenshot, Q quit)')
        print(' 2  Live arm angle + shoulder tilt (A/B/S/Q)')
        print(' 3  Cross-check angles with phone/goniometer')
        print(' 4  Record reference (SPACE once, automatic stop)')
        print(' 5  Record COMPLETE practice repetition')
        print(' 6  Record INCOMPLETE practice repetition')
        print(' 7  Review current reference (skeleton + angle traces)')
        print(' 8  Score practice + suggest DTW threshold')
        print(' 9  Freeze PRACTICE DTW threshold')
        print('10  Record held-out test repetition')
        print('11  Score held-out tests')
        print('12  Live demo (C/I/U label, SPACE once, auto stop)')
        print('13  Shoulder visibility test')
        print('14  SWITCH POSITION: seated / standing')
        print('15  CHANGE countdown and movement timeout')
        print(' 0  Exit')
        print('-' * 68)
        print()
        choice = input('Select an option (0-15): ').strip()
        if choice == '0':
            return
        if choice == '1':
            run('day0_pose.py')
        elif choice == '2':
            run('step2_check_angle.py')
        elif choice == '3':
            run('step3_calibrate.py')
        elif choice == '4':
            name = next_name(s, 'reference', 'ref')
            print('New reference:', position, name)
            run('step4_record.py', 'reference', name)
            path = nf.mode_dir('clips', s) / 'reference' / (name + '.avi')
            if path.exists() and (path.with_suffix('.json')).exists():
                metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                if metadata.get('capture_outcome') != 'complete':
                    print('Reference attempt saved for review. Record a full raise and return to set the reference.')
                    continue
                s['reference_clip'] = path.as_posix()
                s['dtw_threshold'] = None
                s['dtw_calibration'] = None
                nf.save_settings(s)
                # Previous calibration belongs to a different reference.
                traces = scoring_dir(s)
                for filename in ('summary_practice.csv', 'practice_meta.json', 'reference_meta.json'):
                    stale = traces / filename
                    if stale.exists():
                        stale.unlink()
                print(f'New {position} reference saved. DTW reset for {position} ONLY.')
                print('Record 3 complete + 3 incomplete practice clips for this reference.')
            else:
                print('Reference was cancelled; previous settings are unchanged.')
        elif choice == '5':
            if not ref or ref == 'NOT RECORDED':
                print('First record a reference with option 4.')
            else:
                run('step4_record.py', 'practice', next_name(s,'practice','complete'))
        elif choice == '6':
            if not s['reference_clip']:
                print('First record a reference with option 4.')
            else:
                run('step4_record.py', 'practice', next_name(s,'practice','incomplete'))
        elif choice == '7':
            if s['reference_clip'] and Path(s['reference_clip']).exists():
                run('check_clip.py', s['reference_clip'])
            else:
                print(f'No {position} reference yet. Choose 4 first.')
        elif choice == '8':
            run('step5_score.py')
        elif choice == '9':
            value = input('Enter suggested PRACTICE DTW threshold (ENTER cancels): ').strip()
            if value:
                run('step5_score.py', '--freeze', value)
        elif choice == '10':
            if s['dtw_threshold'] is None:
                print('First score practice (8) and freeze a threshold (9).')
                continue
            name = next_name(s, 'test', 'T')
            label = input('Expected label BEFORE recording [complete/incomplete/uncertain]: ').strip().lower()
            if label not in ('complete','incomplete','uncertain'):
                print('Invalid label; test not recorded.')
                continue
            run('step4_record.py', 'test', name)
            if (nf.mode_dir('clips',s) / 'test' / (name+'.json')).exists():
                labels = nf.mode_dir('clips',s) / 'test' / 'expected_labels.csv'
                exists = labels.exists()
                with labels.open('a', newline='', encoding='utf-8') as f:
                    wr = csv.writer(f)
                    if not exists:
                        wr.writerow(['clip','expected_label'])
                    wr.writerow([name,label])
                print('Expected label saved:', position, name, label)
        elif choice == '11':
            run('step5_score.py', 'test')
        elif choice == '12':
            run('step6_demo.py')
        elif choice == '13':
            run('visibility_test.py', 'shoulder')
        elif choice == '14':
            other = 'standing' if position == 'seated' else 'seated'
            selected = input(f'Switch from {position} to {other}? [Y/n]: ').strip().lower()
            if selected in ('','y','yes'):
                nf.set_position(other)
                print(f'Now using {other.upper()} recordings and calibration. The {position} data is preserved.')
        elif choice == '15':
            try:
                cd = input(f'Countdown seconds 3-20 [{s["record_countdown_s"]}]: ').strip()
                dur = input(f'Movement timeout seconds 6-30 [{s["rep_timeout_s"]}]: ').strip()
                nf.set_recording_timing(int(cd) if cd else s['record_countdown_s'],
                                        int(dur) if dur else s['rep_timeout_s'])
                print('Timing saved for both positions.')
            except ValueError as exc:
                print('Invalid timing:', exc)
        else:
            print('Please choose 0-15.')


if __name__ == '__main__':
    os.chdir(PROJECT_DIR)
    main()
