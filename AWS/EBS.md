---
title: "EBS Commands"
author: "Shanaka DeSoysa"
date: 2020-06-08T00:00:00
description: "EBS Commands"
type: technical_note
draft: true
---

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html

# EBS Commands
`lsblk`

`sudo file -s /dev/xvdb`

`sudo mkfs -t xfs /dev/xvdb`

`sudo mkdir /data`

`sudo mount /dev/xvdb /data`

`lsblk`

cp /etc/fstab /etc/fstab.orig

blkid

sudo file -s /dev/xvdb

sudo umount /data

sudo mount -a
