#!/usr/bin/env python

# Usage: python3.8 ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 500

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "/app/"
default_input_dir = root_dir + "data"
default_output_dir = root_dir + "sample_result_submission"
default_program_dir = root_dir + "program/ingestion_program"
default_submission_dir = root_dir + "ingested_program"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 6

# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import sys
import datetime
import json
import importlib
import dill
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

# =========================== BEGIN PROGRAM ================================
def fileExists(path):
    if not os.path.exists(path):
        print(path)
        raise ModelApiError("Missing file : ", path)
        exit_program()   

def import_parameters(submission_dir):
    ## import parameters.json as a dictionary
    path_submission_parameters = os.path.join(submission_dir, 'parameters.json')
    if not os.path.exists(path_submission_parameters):
        raise ModelApiError("Missing parameters.json file")
        exit_program()
    with open(os.path.join(submission_dir, 'parameters.json')) as json_file:
        parameters = json.load(json_file)
    return parameters

def exit_program():
    print("Error exiting")
    sys.exit(0)

class ModelApiError(Exception):
    """Model api error"""

    def __init__(self, msg=""):
        self.msg = msg
        print(msg)


class TimeoutException(Exception):
    """timeoutexception"""


if __name__=="__main__" :
    #### Check whether everything went well (no time exceeded)
    execution_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)

    #### Test with LIPS dataset ####
    print("## Starting Ingestion program ##")

    # import configuration file 
    run_parameters = import_parameters(submission_dir)
    print("Run parameters: ", run_parameters)


    start_total_time = time.time()
    from lips import get_root_path

    LIPS_PATH = get_root_path()
    # dataset recovered from host
    DIRECTORY_NAME = '/app/data/Dataset'
    BENCHMARK_NAME = "Case1"
    LOG_PATH = "/app/lips_logs.log"
    BENCH_CONFIG_PATH = os.path.join("/LIPS","configurations","airfoil","benchmarks","confAirfoil.ini") #Configuration file related to the benchmark
    SIM_CONFIG_PATH = os.path.join(submission_dir,"config.ini")
    SAVE_PATH = os.path.join(output_dir,"AirfRANSModel")
    
    # FIXME: if evaluateonly true : copy results for evaluation, deactivate option for competition phase

    if run_parameters["scoringonly"]:
        print("Scoring only mode activated")
        print("Copying results from submission to output directory")

        # open results file 
        resultpath = os.path.join(submission_dir,"results.json")
        if not os.path.exists(resultpath):
            raise ModelApiError("Missing results.json file")
            exit_program()

        # save evaluation for scoring program
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(resultpath) as json_file:
            results = json.load(json_file)
    
        json_metrics = json.dumps(results, indent=4)
        # Writing to sample.json
        with open(os.path.join(output_dir, 'json_metrics.json'), "w") as outfile:
            outfile.write(json_metrics)
        exit(1)


    # Benchmark loaded through dill to simplify 
    # print("Preparing benchmark")
    # from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
    # # benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
    # #                             config_path = BENCH_CONFIG_PATH,
    # #                             benchmark_name = BENCHMARK_NAME,
    # #                             log_path = LOG_PATH)
    # # benchmark.load(path=DIRECTORY_NAME)  
    dill.load_session("/app/data/benchmark_session")

    

    print("Input attributes (features): ", benchmark.config.get_option("attr_x"))
    print("Output attributes (targets): ", benchmark.config.get_option("attr_y"))

    simulator_parameters = run_parameters["simulator_config"]

    print("Preparing scaler")

        # Legacy submissions
    if "scaler_type" not in simulator_parameters:
        print("Legacy submission detected")
        if simulator_parameters["custom_scaler"] == True:
            simulator_parameters["scaler_type"] = "custom"
            simulator_parameters["scaler_file"] = "my_scaler"
        else:
            simulator_parameters["scaler_type"] = "simple"

    if simulator_parameters["scaler_type"] == "simple":
        print("Loading LIPS scaler " + simulator_parameters["scaler"])
        scaler_module = importlib.import_module("lips.dataset.scaler."+simulator_parameters["scaler_class"])
        scaler_class = getattr(scaler_module, simulator_parameters["scaler"])
        # Import user-provided scaler parameters
        fileExists(os.path.join(submission_dir,"scaler_parameters.py"))
        from scaler_parameters import compute_scaler_parameters
        scalerParams = compute_scaler_parameters(benchmark)
        print("Scaler Parameters")
        print(scalerParams)

    elif simulator_parameters["scaler_type"] == "custom":
        print("Custom scaler")
        print("Loading custom scaler from submission directory")
        fileExists(os.path.join(submission_dir,simulator_parameters["scaler_file"]+".py"))

        ## load custom scaler from submission directory
        scaler_module = importlib.import_module(simulator_parameters["scaler_file"])
        scaler_class = getattr(scaler_module, simulator_parameters["scaler"])

        # Import user-provided scaler parameters
        fileExists(os.path.join(submission_dir,"scaler_parameters.py"))
        from scaler_parameters import compute_scaler_parameters
        scalerParams = compute_scaler_parameters(benchmark)
        print("Scaler Parameters")
        print(scalerParams)
    else:
        print("No scaler specified")
        scaler_class = None
        scalerParams = None


    print("Preparing Simulator")
    
    # Legacy submissions
    if "simulator_type" not in simulator_parameters:
        print("Legacy submission detected")
        if simulator_parameters["custom_simulator"] == True:
            simulator_parameters["simulator_type"] = "custom_torch"
            simulator_parameters["simulator_file"] = "my_augmented_simulator"
        else:
            simulator_parameters["simulator_type"] = "simple_torch"


    if simulator_parameters["simulator_type"] == "simple_torch":
        print("Loading LIPS torch simulator " + simulator_parameters["model"])
        simulator_module = importlib.import_module("lips.augmented_simulators.torch_models."+simulator_parameters["model_type"])
        simulator_class = getattr(simulator_module, simulator_parameters["model"])

        from lips.augmented_simulators.torch_simulator import TorchSimulator
        simulator = TorchSimulator(name=simulator_parameters["name"],
                           model=simulator_class,
                           scaler=scaler_class,
                           scalerParams=scalerParams,
                           log_path="log_benchmark",
                           device="cuda:0",
                           bench_config_path=BENCH_CONFIG_PATH,
                           bench_config_name=BENCHMARK_NAME,
                           sim_config_path=SIM_CONFIG_PATH,
                           sim_config_name=simulator_parameters["config_name"],
                           architecture_type="Classical",
                           **run_parameters["simulator_extra_parameters"]                      
                          )

    elif simulator_parameters["simulator_type"] == "custom_torch": 
        print("Custom torch LIPS simulator")
        print("Loading custom simulator from submission directory")
        ## load custom simulator from submission directory
        fileExists(os.path.join(submission_dir,simulator_parameters["simulator_file"]+'.py'))
        # Import user-provided simulator code
        simulator_module = importlib.import_module(simulator_parameters["simulator_file"])
        simulator_class = getattr(simulator_module, simulator_parameters["model"])
        from lips.augmented_simulators.torch_simulator import TorchSimulator
        simulator = TorchSimulator(name=simulator_parameters["name"],
                           model=simulator_class,
                           scaler=scaler_class,
                           scalerParams=scalerParams,
                           log_path="log_benchmark",
                           device="cuda:0",
                           bench_config_path=BENCH_CONFIG_PATH,
                           bench_config_name=BENCHMARK_NAME,
                           sim_config_path=SIM_CONFIG_PATH,
                           sim_config_name=simulator_parameters["config_name"],
                           architecture_type="Classical",
                           **run_parameters["simulator_extra_parameters"]                      
                          )

    elif simulator_parameters["simulator_type"] == "simple_tf":
        print("Loading LIPS tensorflow simulator " + simulator_parameters["model"])
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
        # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)
        print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
        simulator_module = importlib.import_module("lips.augmented_simulators.tensorflow_models.airfoil."+simulator_parameters["model_type"])
        simulator_class = getattr(simulator_module, simulator_parameters["model"])
        simulator = simulator_class(name=simulator_parameters["name"],
                                bench_config_path=BENCH_CONFIG_PATH,
                                bench_config_name=BENCHMARK_NAME,
                                sim_config_path=SIM_CONFIG_PATH,
                                sim_config_name=simulator_parameters["config_name"],
                                scaler=scaler_class,
                                scalerParams=scalerParams,
                                log_path="log_benchmark")

    elif simulator_parameters["simulator_type"] == "custom":
        print("Loading custom simulator " + simulator_parameters["model"])
        print("Loading custom simulator " + simulator_parameters["simulator_file"])
        ## load custom simulator from submission directory
        fileExists(os.path.join(submission_dir,simulator_parameters["simulator_file"]+'.py'))
        # Import user-provided simulator code
        simulator_module = importlib.import_module(simulator_parameters["simulator_file"])
        simulator_class = getattr(simulator_module, simulator_parameters["model"])

        simulator = simulator_class(benchmark=benchmark,
                                **run_parameters["simulator_extra_parameters"]
                                )


    if run_parameters["evaluateonly"]:
        print("Evaluation only mode activated")
        print("Loading trained model")
        LOAD_PATH = os.path.join(submission_dir,"trained_model")
        simulator.restore(path=LOAD_PATH)
    else:
        print("Training simulator")
        start = time.time()
        simulator.train(benchmark.train_dataset, 
                    save_path=None, 
                    **run_parameters["training_config"]
                    )
        training_time = time.time() - start

        print("Run successfull in " + str(training_time) + " seconds")
        try:
            print("Number of parameters :", simulator.count_parameters())
            print("Summary :")
            simulator.summary()
        except:
            print("Could not count parameters")





    print("Starting evaluation")
    
    start_test = time.time()
    fc_metrics_test = benchmark.evaluate_simulator(dataset="test",augmented_simulator=simulator,eval_batch_size=256000 )
    test_evaluation_time = time.time() - start_test
    test_mean_simulation_time = test_evaluation_time/len(benchmark._test_dataset.get_simulations_sizes())

    start_test_ood = time.time()
    fc_metrics_test_ood = benchmark.evaluate_simulator(dataset="test_ood",augmented_simulator=simulator,eval_batch_size=256000 )
    test_ood_evaluation_time = time.time() - start_test_ood
    test_ood_mean_simulation_time = test_ood_evaluation_time/len(benchmark._test_ood_dataset.get_simulations_sizes())
    
    simulator_metrics = {
        "total_time":time.time() - start_total_time,
        "training_time":training_time,
        "test_evaluation_time":test_evaluation_time,
        "test_mean_simulation_time":test_mean_simulation_time,
        "test_ood_evaluation_time":test_ood_evaluation_time,
        "test_ood_mean_simulation_time":test_ood_mean_simulation_time,
        "fc_metrics_test":fc_metrics_test,
        "fc_metrics_test_ood":fc_metrics_test_ood
    }

    # save evaluation for scoring program
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    json_metrics = json.dumps(simulator_metrics, indent=4)
    # Writing to sample.json
    with open(os.path.join(output_dir, 'json_metrics.json'), "w") as outfile:
        outfile.write(json_metrics)

    print(simulator_metrics)
   