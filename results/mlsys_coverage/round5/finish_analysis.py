"""Wait for all planned GPU work, then regenerate and audit the final artifacts.

This performs analysis only. It does not pick additional parameters, start
experiments, commit, or publish. Failures stop the sequence for inspection.
"""
import subprocess,sys,time
from make_plans import HERE,save
from monitor import snapshot


def main():
    while True:
        status,done=snapshot()
        if done:break
        time.sleep(30)
    scripts=['analyze_screen.py','analyze_final.py','analyze_followups.py',
             'analyze_postopt.py','analyze_breakdown.py','analyze_ssd.py',
             'audit_parameter_effects.py','audit_tuning_tails.py','audit_capture_tails.py','summarize_telemetry.py','audit_study.py',
             'make_figs.py','make_report.py']
    records=[]
    for name in scripts:
        print('RUN',name,flush=True)
        log=HERE/('screen_progress.txt' if name=='analyze_screen.py' else name[:-3]+'.log')
        with log.open('w') as handle:
            subprocess.run([sys.executable,str(HERE/name)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        records.append(dict(script=name,log=log.name,status='complete'))
        print('DONE',name,flush=True)
    save('ANALYSIS_COMPLETE.json',records)


if __name__=='__main__':main()
