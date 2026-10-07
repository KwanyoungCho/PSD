"""Restore into an empty temporary checkout, validate bytes and path relocation."""
import importlib.util,json,os,shutil,subprocess,sys,tempfile
from make_plans import HERE,save
from analyze_screen import analyze
from result_metrics import read


def main():
    manifest=read(HERE/'ARCHIVE_MANIFEST.json')
    with tempfile.TemporaryDirectory(prefix='duet_round5_restore_') as folder:
        from pathlib import Path
        root=Path(folder)
        for name in ('archive_results.py','ARCHIVE_MANIFEST.json','analyze_screen.py','EXCLUSIONS.json'):
            shutil.copy2(HERE/name,root/name)
        os.symlink(HERE/'archives',root/'archives',target_is_directory=True)
        # Every raw destination is absent: this exercises decompression,
        # directory creation, SHA checks and mtime restoration in the actual
        # shipped restore command, not just existing-file validation.
        subprocess.run([sys.executable,str(root/'archive_results.py'),'--restore'],check=True)
        for entry in manifest:
            path=root/entry['path']
            if path.stat().st_mtime_ns!=entry['mtime_ns']:raise ValueError('Restored mtime differs')
            if path.stat().st_size!=entry['bytes']:raise ValueError('Restored size differs')
        spec=importlib.util.spec_from_file_location('relocated_round5_screen',root/'analyze_screen.py')
        relocated=importlib.util.module_from_spec(spec);spec.loader.exec_module(relocated)
        checks=[]
        for name in ('llama3_b8_refine_retry/llama3_b8_refine_e30_k3_2_r2.json',
                     'llama3_stability_l1_profile/llama3_b8_stability_profile_4.json'):
            a,b=analyze(HERE/name),relocated.analyze(root/name)
            if json.dumps(a,sort_keys=True)!=json.dumps(b,sort_keys=True):raise ValueError('Relocated diagnostic differs')
            checks.append(dict(path=name,diagnostic_equal=True,profile_paths=b['profile_paths']))
        result=dict(status='passed',fresh_destination=True,files=len(manifest),
            raw_bytes=sum(x['bytes'] for x in manifest),sha256_verified_by_restore=True,
            all_mtimes_equal=True,all_sizes_equal=True,relocated_diagnostics=checks,
            temporary_copy_removed_on_exit=True)
    save('ARCHIVE_RESTORE_AUDIT.json',result)
    print('Fresh restore and relocated diagnostics passed:',len(manifest),'files')


if __name__=='__main__':main()
