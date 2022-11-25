---
title: "EC2 Simple Web Server"
author: "Shanaka DeSoysa"
date: 2022-11-24
subtitle: Last updated on 2022-11-24
description: "EC2 Simple Web Server bootstrap script"
type: technical_note
keywords: aws, ec2, webserver
---

# Run a Simple Web Server

Use following script in the User Data section of the EC2 instance configuration.

```bash
#!/bin/bash
# Use this for your user data (script from top to bottom)
# install httpd (Linux 2 version)
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd
echo "<h1>Hello World from $(hostname -f)</h1>" > /var/www/html/index.html
```
