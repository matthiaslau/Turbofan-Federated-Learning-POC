#!/bin/sh

set -e

notebook_pod=`kubectl get pods | awk '{print $1}' | grep 'notebooks'`
trainer_pod=`kubectl get pods | awk '{print $1}' | grep 'trainer'`

#echo "Loading turbofan data to pod $notebook_pod..."

#kubectl exec $notebook_pod mkdir /data
#kubectl exec $notebook_pod mkdir /models

#kubectl cp ../data/train_data_initial.txt $notebook_pod:/data/train_data_initial.txt
#kubectl cp ../data/test_data_test.txt $notebook_pod:/data/test_data_test.txt
#kubectl cp ../data/test_data_val.txt $notebook_pod:/data/test_data_val.txt

echo "Loading data and initial model to pod $trainer_pod..."

kubectl exec $trainer_pod mkdir /data
kubectl exec $trainer_pod mkdir /models

kubectl cp ../data/test_data_test.txt $trainer_pod:/data/test_data_test.txt
kubectl cp ../data/test_data_val.txt $trainer_pod:/data/test_data_val.txt

kubectl cp ../models/turbofan_initial.pt $trainer_pod:/models/turbofan_initial.pt

echo "Done."
