#!/usr/bin/zsh

jid1=$(sbatch bashscripts/tabletransformer_train_titw_aachen_callinner.sh 50 | awk '{print $NF}')
echo "$jid1"
jid2=$(sbatch -d afterok:"$jid1" bashscripts/tabletransformer_train_titw_aachen_callinner.sh 100 titw_call_e250_end | awk '{print $NF}')
echo "$jid2"
jid3=$(sbatch -d afterok:"$jid2" bashscripts/tabletransformer_train_titw_aachen_callinner.sh 150 titw_call_e250_end | awk '{print $NF}')
echo "$jid3"
jid4=$(sbatch -d afterok:"$jid3" bashscripts/tabletransformer_train_titw_aachen_callinner.sh 200 titw_call_e250_end | awk '{print $NF}')
echo "$jid4"
jid5=$(sbatch -d afterok:"$jid4" bashscripts/tabletransformer_train_titw_aachen_callinner.sh 250 titw_call_e250_end | awk '{print $NF}')
echo "$jid5"
