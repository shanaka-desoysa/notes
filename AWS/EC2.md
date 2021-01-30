---
title: "EC2 Commands"
author: "Shanaka DeSoysa"
date: 2020-06-08T00:00:00
description: "EC2"
type: technical_note
draft: true
---

## SSH
`ssh -i EC2Tutorial.pem ec2-user@PUBLIC_IP`

## Elevate Rights to Root
sudo su

### Update packages
yum update -y

### Install httpd
yum install -y httpd.x86_64

### Start Service
systemctl start httpd.service

### Enable across reboots
systemctl enable httpd.service

### Check webserver running
curl localhost:80

### Write Default webpage
"Hello World $(hostname -f)" > /var/www/html/index.html

## EC2 User Data
Bootstrap scripts. Under Configure Instance Details > Advanced Details

```bash
#!/bin/bash
# Use this for your user data (script without newlines)
# install httpd (Linux 2 version)
yum update -y
yum install -y httpd.x86_64
systemctl start httpd.service
systemctl enable httpd.service
echo "Hello World from $(hostname -f)" > /var/www/html/index.html
```

