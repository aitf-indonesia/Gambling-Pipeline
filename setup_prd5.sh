#!/bin/bash
set -e

echo "[1] Install Supervisor"
source ~/.bashrc
sudo apt update
sudo apt install -y supervisor

echo "[2] Create Supervisor config (prd5_api)"

sudo tee /etc/supervisor/conf.d/prd5_api.conf > /dev/null <<'EOF'
[program:prd5_api]
command=/home/ubuntu/miniconda3/envs/prd5/bin/python /home/ubuntu/tim5_prd_workdir/Gambling-Pipeline/run_server.py
directory=/home/ubuntu/tim5_prd_workdir/Gambling-Pipeline
autostart=true
autorestart=true
stderr_logfile=/var/log/prd5_api.err.log
stdout_logfile=/var/log/prd5_api.out.log
EOF

echo "[3] Reload Supervisor"
sudo /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
sudo supervisorctl reread
sudo supervisorctl update

echo "[4] Start prd5_api"
sudo supervisorctl start prd5_api || sudo supervisorctl restart prd5_api

echo "Done!"
sudo supervisorctl status