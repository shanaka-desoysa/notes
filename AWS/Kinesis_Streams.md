---
title: "EC2 Commands"
author: "Shanaka DeSoysa"
date: 2020-06-08T00:00:00
description: "EC2"
type: technical_note
draft: true
---

`aws kinesis list-streams`

`aws kinesis describe-stream --stream-name my-first-stream`

# Kinesis Put Record

`aws kinesis put-record --stream-name my-first-stream --data $(echo -n 'user-signup' | base64) --partition-key user_123`
`aws kinesis put-record --stream-name my-first-stream --data $(echo -n 'user-login' | base64) --partition-key user_123`

## 
aws kinesis get-shard-iterator --stream-name my-first-stream --shard-id shardId-000000000000 --shard-iterator-type TRIM_HORIZON

"ShardIterator": "AAAAAAAAAAGUfDi2WFV5Rl1wJhp8r5tNcyVdSRKaSI57nXHY3MroLWBpws9Dep00Mxv6blQDCPbL+8INJJg1GbORybBXKHtsXNJ8nyQv3eQVfNqbaZoxEusDBvdcn6BIMXrgRKfDhZ410NV/qblF2R0DJE2XxAIueXKD9avrIZp459XoI0h+1GG1PEaHP66/FwC1kUf5ruYz4MiYRw0J9hNc3Y43vwsV"

aws kinesis get-records --shard-iterator "AAAAAAAAAAGUfDi2WFV5Rl1wJhp8r5tNcyVdSRKaSI57nXHY3MroLWBpws9Dep00Mxv6blQDCPbL+8INJJg1GbORybBXKHtsXNJ8nyQv3eQVfNqbaZoxEusDBvdcn6BIMXrgRKfDhZ410NV/qblF2R0DJE2XxAIueXKD9avrIZp459XoI0h+1GG1PEaHP66/FwC1kUf5ruYz4MiYRw0J9hNc3Y43vwsV"