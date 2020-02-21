# Turbofan POC: Predictive Maintenance of Turbofan Engines using Federated Learning 

This repository shows a proof of concept (POC) of preventing machine outages using federated learning to continuously 
improve predictions of the remaining lifetime of aircraft gas turbine engines.

![Engine Crash](https://media.giphy.com/media/4OGPHOwyp6MO4/giphy.gif)

For the engine emulation the "Turbofan 
Engine Degradation Simulation Data Set" from the NASA [1] is used. :rocket: The implementation is based on 
[PySyft](https://github.com/OpenMined/PySyft), an awesome library for encrypted, privacy preserving machine learning.

## The Use Case

In this proof of concept the goal is to maintain aircraft gas turbine engines in time before they fail as failures of 
these engines are very expensive for the operating company. To achieve this we will predict the remaining 
useful life (RUL) for the engines and switch them into maintenance a few cycles before we think a failure will
happen.

This task is aggravated by the fact that this use case is from the perspective of the manufacturer of the engines who 
are selling them and has no direct access to the engines operating data as the operating companies consider this data 
as confidential. The manufacturer still wants to offer the described failure early warning system.

There is some data available to the manufacturer from internal turbofan engines that will be used for training an 
initial machine learning model. All engines on the market will be expanded by a software component reading in the sensor 
measurements of the engine and predicting the RUL using this model and reacting on a low RUL with performing a 
maintenance. During a maintenance the theoretical moment of failure will be estimated by the maintenance staff, in this 
proof of concept the engine will be set in maintenance mode and the emulation data will continue to figure out the 
moment of failure. Complete data series up to a failure will then be used the regularly re-train the model to improve 
prediction quality over time.  

## The Data

![Simplified diagram of turbofan engine [2]](images/engine_diagram.png)

<sup>*Simplified diagram of turbofan engine [2]*</sup>

The NASA dataset contains data on engine degradation that was simulated using C-MAPSS (Commercial Modular 
Aero-Propulsion System Simulation). Four different sets were simulated under different combinations of operational 
conditions and fault modes. Each set includes operational settings and sensor measurements (temperature, pressure, fan 
speed, etc.) for several engines for every cycle of their lifetime. For more information on the data see [2].

## Prerequisites

To emulate turbofan engines that can continue to run after a failure / maintenance we are combining multiple engine 
data series from the dataset to one set for each of our engine nodes. These series are then replayed by the engine 
nodes in sequence.

To prepare the data for our POC it needs to be downloaded and split. We will work with the set "FD001" containing 100 
engines for training and 100 engines for validation/testing. The train data is split into one subset for initial 
training (5 engine series) and 5 subsets for each of our engine nodes (19 engines series each). The test data is split 
into one subset for cross-validation (50 engine series) and one subset for evaluation (50 engine series).  

The `data_preprocessor` script is accomplishing all this for us. Ensure you have all requirements installed.

```
python data_preprocessor.py --turbofan_dataset_id=FD001 --engine_percentage_initial=5 --engine_percentage_val=50 --worker_count=5
```

## Data Analysis

Now the project officially begins! :rocket: The first step is analysing the initial data we have centrally as the 
manufacturer to learn more about the data itself. See the [data analysis notebook](notebooks/data_analysis.ipynb).

## Initial Training

The next step is to prepare the data for training and to design a model. Then an initial model is trained, evaluated 
and saved into the model directory. See the [initial training notebook](notebooks/initial_training.ipynb).

## Start the Engines

Let´s start the engines! :cyclone: There is a full setup prepared using docker in the `docker-compose.yml`. It contains 
- a container for **jupyter notebooks**
- a [PyGrid](https://github.com/OpenMined/PyGrid) **grid gateway**
- 5 **engines**
- a **federated trainer**

The engine container consist of a custom engine node and a 
[PyGrid](https://github.com/OpenMined/PyGrid) grid node. The **engine node** is reading in the sensor data, controlling 
the engine state and predicting the RUL using the current model in the grid. The **federated trainer** is regularly 
checking the grid for enough new data and then starting a new federated learning round. After the round is finished the 
new model is served to the grid to be directly used by the engine nodes.

```
docker-compose up -d
```

The engine nodes expose an interface showing the engines state, stats and sensor values: **localhost:800[1-5]**. You can 
also checkout the interface of the grid node: **localhost:300[1-5]**. 

![Engine Node](images/engine_node.jpg)

Also checkout the logs of the federated trainer to see the federated training in action:

```
docker logs -f trainer
```

## Pimp the Engines

There are a lot of parameters in the `docker-compose.yml` and for serious results you need to adjust some of them. Here 
are the most important ones explained:

- **CYCLE_LENGTH (engine)**: The amount of seconds one single engine cycle will take. Decrease to speed up the engine 
emulation.  
- **NEW_DATA_THRESHOLD (trainer)**: The federated trainer will wait for this amount of new data before starting a new 
training round. Increase to prevent training rounds with too few data.
- **EPOCHS (trainer)**: The number of epochs the federated trainer is using for training.

## Bonus: Local Setup

If you want to run the POC locally without docker, no problem. You can start all the nodes manually on your machine. 

Ensure you have all requirements installed and then start with launching the grid gateway. Checkout the 
[PyGrid](https://github.com/OpenMined/PyGrid) repository and go into the gateway directory. Then start the gateway like 
this:

```
python gateway.py --start_local_db --port=5000
```

Next we need to start the grid nodes from the ./engine/grid_node directory. You can also use the grid node from your 
PyGrid repository but check the current parameters from the version you checked out. Execute the following command for 
every engine you want to have in your setup, of course changing the id and port.

```
python websocket_app.py --start_local_db --id=worker1 --port=3000 --gateway_url=http://localhost:5000
```

Now the whole grid setup is working and waiting for commands so let´s continue launching the engines from the 
./engine/engine_node directory:

```
python turbofan_worker.py --id=engine1 --port=8000 --grid_node_address=localhost:3000 --grid_gateway_address=localhost:5000 --data_dir=../../data --dataset_id=1 --cycle_length=1
``` 

Launch as many engines as grid nodes you have, connecting them with the `grid_node_address` parameter.

The engines directly start running, you can check this out on their web interface (http://localhost:8000). So the only 
piece still missing is the federated trainer. You can start it like this:

```
python federated_trainer.py --grid_gateway_address=localhost:5000 --new_data_threshold=250 --scheduler_interval=10 --epochs=70 --data_dir=../data --model_dir=../models
```

The setup is now ready and running! :tada:

### Building docker images

If you want to build your own docker images with code changes you can easily use the `build-docker-images.sh` script. 
It will use a base image from docker hub with main dependencies to build an image for the engine, an image for the 
federated trainer and an image for the jupyter notebook environment.

## Join the PySyft Community!

Join the rapidly growing PySyft community of 5000+ on [Slack](http://slack.openmined.org). The slack community is very 
friendly and great about quickly answering questions about the use and development of PySyft!

## References

[1] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data 
Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research 
Center, Moffett Field, CA

[2] Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, 
https://data.nasa.gov/dataset/Damage-Propagation-Modeling-for-Aircraft-Engine-Ru/j94x-wgir
