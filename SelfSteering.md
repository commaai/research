# Steering Angle model
Follow the instructions to train a deep neural network for self-steering cars.
This experiment is similar to [End to End Learning for Self-Driving
Cars](https://arxiv.org/abs/1604.07316).

1) Download dataset
```
./get_data.sh
```

2) Start training data server in the first terminal session
```bash
./server.py --batch 200 --port 5557
```  

3) Start validation data server in a second terminal session
```bash
./server.py --batch 200 --validation --port 5556
```

4) Train steering model in a third terminal
```bash
./train_steering_model.py --port 5557 --val_port 5556
```

4) Visualize results
```bash
./view_steering_model.py ./outputs/steering_model/steering_angle.json
```
<img src="./images/selfsteer.gif">

Your job is to make the predicted `green` path to be equal to the actual `blue` path. Once you get that you have a self-driving car. Next step: get yourself some funding.

