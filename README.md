## Project structure

+ [/cboosted](https://github.com/zexUlt/spiked-neural-network-py/tree/main/cboosted) -- my attempts to use Cython for performance boost in some calculations (no luck)
+ [/compiled](https://github.com/zexUlt/spiked-neural-network-py/tree/main/compiled) -- another attempts to increase performance but using Numba package and precompile some modules before use (same, no luck)
+ [/data](https://github.com/zexUlt/spiked-neural-network-py/tree/main/data) -- directory with sample training data packed into .npy archives. `vl` and `tr` prefixes stand for `validation` and `training` and contain the same data as in files without those prefixes
+ [/models](https://github.com/zexUlt/spiked-neural-network-py/tree/main/models) -- directory where models are located. Newer version is in [/models/v2](https://github.com/zexUlt/spiked-neural-network-py/tree/main/models/v2)
+ [/playground](https://github.com/zexUlt/spiked-neural-network-py/tree/main/playground) -- directory for hypothesis-checking scripts
+ [activation_functions.py](https://github.com/zexUlt/spiked-neural-network-py/tree/main/activation_functions.py) -- module containing various activation functions. Available functions:
  + `Izhikevich` -- Izhikevich artificial neuron activation function
  + `Sigmoid`
+ [data_generator.py](https://github.com/zexUlt/spiked-neural-network-py/tree/main/data_generator.py) -- generates data using some producer
+ [data_producer.py](https://github.com/zexUlt/spiked-neural-network-py/tree/main/data_producer.py) -- module with data producers by some predefined law. Available producers:
  + `IzhikevichProducer` -- Izhikevich neuron-like signal producer
  + `CallableProducer` -- produces data using given callable object
+ [gamma_func.py](https://github.com/zexUlt/spiked-neural-network-py/tree/main/gamm_func.py) -- module with special distance-dependent functions. Available functions:
  + `MultiplicativeGamma` -- gamma-function with free parameter multiplied by distance
  + `PowerGamma` -- gamma-functuion with free parameter as power of distance
+ [projectors.py](https://github.com/zexUlt/spiked-neural-network-py/tree/main/projectors.py) -- module with various vector projectors. Available projectors:
  + `NullProjector` a.k.a. `IdentityProjector` -- the vector as is
  + `SaturatedProjector` a.k.a. `BoxProjector` -- projects given vector inside predefined box
  + `EllipsoidProjector` -- projects given vector inside predefined ellipse
+ [requirements.txt](https://github.com/zexUlt/spiked-neural-network-py/tree/main/requirements.txt) -- may or may not be insufficient. If it is insufficient, please, update it. Thank you!
