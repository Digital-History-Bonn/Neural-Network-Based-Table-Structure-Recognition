#!/usr/bin/zsh

jid1=$(sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:1 bashscripts/tabletrain_gputest_inner.sh 1 1 | awk '{print $NF}')
echo "$jid1"
jid2=$(sbatch --nodes=1 --ntasks-per-node=2 --gres=gpu:2 -d afterok:"$jid1" bashscripts/tabletrain_gputest_inner.sh 1 2 | awk '{print $NF}')
echo "$jid2"
jid3=$(sbatch --nodes=1 --ntasks-per-node=3 --gres=gpu:3 -d afterok:"$jid2" bashscripts/tabletrain_gputest_inner.sh 1 3 | awk '{print $NF}')
echo "$jid3"
jid4=$(sbatch --nodes=1 --ntasks-per-node=4 --gres=gpu:4 -d afterok:"$jid3" bashscripts/tabletrain_gputest_inner.sh 1 4 | awk '{print $NF}')
echo "$jid4"
jid5=$(sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:4 -d afterok:"$jid4" bashscripts/tabletrain_gputest_inner.sh 2 4| awk '{print $NF}')
echo "$jid5"
