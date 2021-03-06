#!/usr/bin/env python

import boto.sqs
from boto.sqs.message import Message
import sys, os

conn = boto.sqs.connect_to_region("us-east-1")
q = conn.get_queue('numbers')
m = boto.sqs.message.Message()

if len(sys.argv) > 1 and sys.argv[1]:
    game_num = int(sys.argv[1])
    m.set_body(str(game_num))
    q.write(m)
    exit()

FIRST_NUM=int(os.environ['FIRST_NUM'])
LAST_NUM=int(os.environ['LAST_NUM'])

batch = []
for game_num in range(FIRST_NUM,LAST_NUM+1):

    m.set_body(str(game_num))
    batch.append((game_num, m.get_body_encoded(), 0))
    if len(batch) == 10:
        q.write_batch(batch)
        batch = []
    if game_num % 100 == 0:
        print('.', end=' ')
        sys.stdout.flush()
