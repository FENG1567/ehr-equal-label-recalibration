#!/usr/bin/env python3
"""Refresh package checksums after metadata edits, refusing changed scientific inputs."""
from pathlib import Path
from reproduce import ROOT,sha,verify
def main():
    verify()
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.sha256'
           and not any(x in {'.git','.venv','outputs','__pycache__'} for x in p.relative_to(ROOT).parts)]
    (ROOT/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT)).replace('\\','/')+'\n' for p in files),encoding='utf-8')
    print(f'REFRESHED {len(files)} package checksums; scientific input hashes unchanged')
if __name__=='__main__':main()
