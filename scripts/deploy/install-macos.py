#!/usr/bin/env python3
"""Install user LaunchAgents; no root required. --uninstall stops these two agents only."""
import argparse,os,pathlib,plistlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--config-dir',required=True);p.add_argument('--uninstall',action='store_true');a=p.parse_args()
exe=pathlib.Path(a.executable).resolve();config=pathlib.Path(a.config_dir).resolve();target=pathlib.Path.home()/'Library/LaunchAgents';target.mkdir(exist_ok=True)
logs=config/'logs';logs.mkdir(exist_ok=True);domain='gui/'+str(os.getuid())
if not a.uninstall and not exe.is_file():raise SystemExit('Executable is missing')
for name,command,filename in [('coordinator','coordinator','coordinator.json'),('metal','split-serve','worker.json')]:
 label='com.infernet.'+name;path=target/(label+'.plist')
 if a.uninstall:
  subprocess.run(['launchctl','bootout',domain+'/'+label],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  path.unlink(missing_ok=True);continue
 if path.exists():raise SystemExit(str(path)+' already exists; uninstall the old release first (drain active requests before upgrading).')
 if not (config/filename).is_file():raise SystemExit('Missing '+filename)
 data={'Label':label,'ProgramArguments':[str(exe),command,'--config',str(config/filename)],'WorkingDirectory':str(config),'RunAtLoad':True,'KeepAlive':{'SuccessfulExit':False},'ThrottleInterval':10,'StandardOutPath':str(logs/(name+'.stdout.log')),'StandardErrorPath':str(logs/(name+'.stderr.log'))}
 path.write_bytes(plistlib.dumps(data));path.chmod(0o600);subprocess.run(['launchctl','bootstrap',domain,str(path)],check=True)
 print('Installed',label)
