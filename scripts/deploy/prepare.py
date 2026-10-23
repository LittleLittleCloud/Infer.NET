#!/usr/bin/env python3
"""Create private deployment configuration and pinned certificates. Never commit this output."""
import argparse,hashlib,json,os,pathlib,secrets,subprocess
p=argparse.ArgumentParser()
p.add_argument('--output',required=True);p.add_argument('--mac-address',required=True)
p.add_argument('--mac-weights',required=True);p.add_argument('--windows-weights',required=True)
p.add_argument('--windows-config-dir',required=True)
a=p.parse_args();os.umask(0o077);root=pathlib.Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
token=secrets.token_hex(32);password=secrets.token_hex(32);pins={}
for role in ['mac','windows']:
 d=root/role;d.mkdir();(d/'token.txt').write_text(token+'\n');(d/'password.txt').write_text(password+'\n')
 conf=d/'certificate.cnf';conf.write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=Infer.NET '+role+'\n[ext]\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth,clientAuth\n')
 subprocess.run(['openssl','req','-x509','-newkey','rsa:3072','-nodes','-days','365','-keyout',str(d/'key.pem'),'-out',str(d/'cert.pem'),'-config',str(conf)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 subprocess.run(['openssl','pkcs12','-export','-out',str(d/'identity.pfx'),'-inkey',str(d/'key.pem'),'-in',str(d/'cert.pem'),'-passout','file:'+str(d/'password.txt')],check=True)
 der=subprocess.check_output(['openssl','x509','-in',str(d/'cert.pem'),'-outform','DER']);pins[role]=hashlib.sha256(der).hexdigest()
 (d/'key.pem').unlink();conf.unlink()
mac=root/'mac';win=a.windows_config_dir.rstrip('\\/')
common={'kv':'fp32','deadline-seconds':'300','queue-size':'2'}
mc={**common,'listen':'http://127.0.0.1:5201','coordinator':'http://127.0.0.1:5200','data-listen':'tcp://'+a.mac_address+':5210','allow-remote':True,'weights':str(pathlib.Path(a.mac_weights).resolve()),'token-file':str(mac/'token.txt'),'tls-cert':str(mac/'identity.pfx'),'tls-password-file':str(mac/'password.txt'),'tls-peer-sha256':pins['windows'],'lock-file':str(mac/'metal.lock')}
wc={'kv':'fp32','cuda-prefill-chunk':'1024','deadline-seconds':'300','endpoint':'tcp://'+a.mac_address+':5210','weights':a.windows_weights,'token-file':win+'\\token.txt','tls-cert':win+'\\identity.pfx','tls-password-file':win+'\\password.txt','tls-peer-sha256':pins['mac'],'lock-file':win+'\\cuda.lock'}
for file,data in [(mac/'worker.json',mc),(mac/'coordinator.json',{'listen':'http://127.0.0.1:5200','token-file':str(mac/'token.txt')}),(root/'windows'/'worker.json',wc)]:file.write_text(json.dumps(data,indent=2)+'\n')
print('Private configuration created:',root)
print('Transfer only windows/ to the configured Windows directory, then restrict its ACL to the service account, SYSTEM and Administrators.')
