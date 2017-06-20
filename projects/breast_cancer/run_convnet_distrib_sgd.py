from systemml import dml, MLContext

ml = MLContext(sc)
ml.setStatistics(True)
ml.setExplain(True)
ml.setExplainLevel("hops")
sc.setLogLevel("WARN")  #INFO")

script = """
# Imports
source("breastcancer/convnet_distrib_sgd.dml") as clf

# Hyperparameters & Settings
lr = 0.001  # learning rate
mu = 0.9  # momentum
decay = 0.99  # learning rate decay constant
lambda = 0.00385
batch_size = 32
parallel_batches = 19
N = batch_size * parallel_batches * 2
epochs = 1
log_interval = 1
dir = "models/lenet-cnn/train/"

# Dummy data
[X, Y, C, Hin, Win] = clf::generate_dummy_data(N)
[X_val, Y_val, C, Hin, Win] = clf::generate_dummy_data(100)

# Train
[Wc1, bc1, Wc2, bc2, Wc3, bc3, Wa1, ba1, Wa2, ba2] =
    clf::train(X, Y, X_val, Y_val, C, Hin, Win, lr, mu, decay,
               lambda, batch_size, parallel_batches, epochs,
               log_interval, dir)

"""
outputs = ("Wc1", "bc1", "Wc2", "bc2", "Wc3", "bc3", "Wa1", "ba1", "Wa2", "ba2")
dml_script = dml(script).output(*outputs)
outs = ml.execute(dml_script).get(*outputs)
outs

