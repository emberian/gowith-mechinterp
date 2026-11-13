#!/usr/bin/env python3
"""Restore hash-pinned saved runs from this repository's optional evidence ref."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def oid(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['restore','verify','list']);p.add_argument('--only',default='');a=p.parse_args();index=json.loads((ROOT/'.run-data/index.json').read_text());entries=[e for e in index['entries'] if not a.only or e['path']==a.only or e['path'].startswith(a.only.rstrip('/')+'/')]
 if not entries:raise ValueError('No matching run data')
 if a.action=='list':
  for e in entries:print(e['path']+'\t'+str(e['size'])+'\t'+e['oid'])
  return
 if a.action=='restore' and subprocess.run(['git','-C',str(ROOT),'cat-file','-e',index['commit']+'^{commit}'],capture_output=True).returncode:
  print('Fetching preserved runs from origin.',file=sys.stderr);subprocess.run(['git','-C',str(ROOT),'fetch','--no-tags','origin',index['ref']],check=True)
  if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','FETCH_HEAD'],text=True).strip()!=index['commit']:raise ValueError('Fetched run archive differs from its pinned commit')
  subprocess.run(['git','-C',str(ROOT),'update-ref',index['ref'],index['commit']],check=True)
 counts={'matching':0,'restored':0,'missing':0}
 for e in entries:
  path=Path(e['path'])
  if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe run path')
  dest=ROOT/path
  for parent in dest.parents:
   if parent==ROOT:break
   if parent.is_symlink():raise ValueError('Symlink parent: '+str(parent))
  if dest.exists() or dest.is_symlink():
   if dest.is_symlink() or not dest.is_file() or oid(dest.read_bytes())!=e['oid']:raise ValueError('Refusing to overwrite different run data: '+e['path'])
   counts['matching']+=1;continue
  if a.action=='verify':counts['missing']+=1;continue
  raw=subprocess.check_output(['git','-C',str(ROOT),'show',index['commit']+':'+e['path']])
  if len(raw)!=e['size'] or oid(raw)!=e['oid']:raise ValueError('Archived run bytes mismatch')
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(raw)
  dest.chmod(0o755 if e['mode']=='100755' else 0o644);counts['restored']+=1
 print(json.dumps({'paths':len(entries),**counts}))
 if counts['missing']:raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except ValueError as exc:raise SystemExit(str(exc))
