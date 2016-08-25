Manual
======
.. role:: bash(code)
    :language: bash


In the following we will show how to use **SMAC3**.

.. note::

    TODO:
        * Miniexample
        * More complex example like Spear-qcp
        * Python Wrapper (basically annotate one of the examples [leadingones has categoricals])

.. _quick:

Quick Start
-----------
| If you haven't installed *SMAC* yet take a look at the `installation instructions <installation.html>`_ and make sure that all the requirements are fulfilled.
| In the examples folder you can find examples that illustrate how to write scenario files that allow you to use *SMAC* to automatically configure an algorithm for you, as well as examples that show how to directly use *SMAC* in python.

Spear-QCP
_________
| For this example we use *SMAC* to optimize `Spear <http://www.domagoj-babic.com/index.php/ResearchProjects/Spear>`_ on a small subset of the QCP-dataset.
| In *SMACs* root-directory type:

.. code-block:: bash

    cd examples/spear_qcp && ls -l

| In this folder you'll see the following files:
|  **features.txt**:
|    The feature file is contains the features for each instance in a csv-format.

     +--------------------+--------------------+--------------------+-----+
     |      instance      | name of feature 1  | name of feature 2  | ... |
     +====================+====================+====================+=====+
     | name of instance 1 | value of feature 1 | value of feature 2 | ... |
     +--------------------+--------------------+--------------------+-----+
     |         ...        |          ...       |          ...       | ... |
     +--------------------+--------------------+--------------------+-----+
|
|  **instances.txt**
|    The instance file contains the names of all instances one might want to consider during the optimization process.
|
|  **scenario.txt**
|    The scenario file contains all the necessary information about the configuration scenario at hand. A more indepth description about the different options in a scenario file can be found below.
|
|  **run.sh**
|     A shell script calling *SMAC* with the following command:
|     :bash:`python ../../scripts/smac --scenario scenario.txt --verbose DEBUG`
|     This runs *SMAC* with the scenario options specified in the scenario.txt file.
|
| The directory **target_algorithms** contains the wrapper and the executable for Spear and the **instances** folder contains the instances on which *SMAC* will configure Spear.

To run the example type one of the two commands below into a terminal:

.. code-block:: bash

    bash run.sh
    python ../../scripts/smac --scenario scenario.txt --verbose DEBUG

| *SMAC* will run for a few seconds and generate a lot of logging output.
| After *SMAC* finished the configuration process you'll get some final statistics about the configuration process:

.. code-block:: bash

    DEBUG:root:Remaining budget: -11.897580 (wallclock), inf (ta costs), inf (target runs)
    INFO:Stats:##########################################################
    INFO:Stats:Statistics:
    INFO:Stats:#Target algorithm runs: 28
    INFO:Stats:Used wallclock time: 21.90 sec
    INFO:Stats:Used target algorithm runtime: 15.72 sec
    INFO:Stats:##########################################################
    INFO:SMAC:Final Incumbent: Configuration:
      sp-clause-activity-inc, Value: 0.956325431976
      sp-clause-decay, Value: 1.77371504106
      sp-clause-del-heur, Value: 2
      sp-first-restart, Value: 52
      sp-learned-clause-sort-heur, Value: 13
      sp-learned-clauses-inc, Value: 1.12196861555
      sp-learned-size-factor, Value: 0.760013050806
      sp-max-res-lit-inc, Value: 0.909236510144
      sp-max-res-runs, Value: 3
      sp-orig-clause-sort-heur, Value: 1
      sp-phase-dec-heur, Value: 6
      sp-rand-phase-dec-freq, Value: 0.0001
      sp-rand-phase-scaling, Value: 0.825118640774
      sp-rand-var-dec-freq, Value: 0.05
      sp-rand-var-dec-scaling, Value: 1.05290899107
      sp-res-cutoff-cls, Value: 5
      sp-res-cutoff-lits, Value: 1378
      sp-res-order-heur, Value: 6
      sp-resolution, Value: 1
      sp-restart-inc, Value: 1.84809841772
      sp-update-dec-queue, Value: 1
      sp-use-pure-literal-rule, Value: 0
      sp-var-activity-inc, Value: 1.00507435273
      sp-var-dec-heur, Value: 4
      sp-variable-decay, Value: 1.91690063007


The first line shows why *SMAC* terminated. The wallclock time-budget is exhausted. The target algorithm runtime (ta cost) and target algorithm runs were not exhausted since the budget for these were not specified and thus defaulted to infinity.

The statistics further show the used wallclock time, target algorithm runtime and the number of executed target algorithm runs.

| In directory in which you invoked *SMAC* now contain a new folder called **SMAC3-output_YYYY-MM-DD_HH:MM:SS** as well as a file called **target_algo_run.json**.
| The .json file contains the information about the target algorithms *SMAC* just executed. In this file you can see the *status* of the algorithm run, *misc*, the *instance* on which the algorithm was evaluated, which *seed* was used, how much *time* the algorithm needed and with which *configuration* the algorithm was run.
| In the folder *SMAC* generates a file for the runhistory, and two files for the trajectory.