From the k8s directory:

Clean current deployment:
```
kubectl delete -f .
```

Deploy services:
```
kubectl create -f .
```

Check if notebooks and trainer pod are started completely:
```
kubectl get pods
``` 

Push data to pods:
```
./push_data.sh
```
