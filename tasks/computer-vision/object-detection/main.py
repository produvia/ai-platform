"""
Download and create a YOLOv3 Keras model and save it to file and 
load yolov3 model and perform object detection.

Edited from the ai-platform object detection. Allows for image, video and passing of parameters from the main to subsequent entry points
"""

import sys
import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_id, run_info.status))
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % previous_version, git_commit)
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" %
              (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" %
          (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=False)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow(**parameters): 
    #Arguments to the workflow allow parameters to be passed from the main to other entry points
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        # os.environ['SPARK_CONF_DIR'] = os.path.abspath('.')
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        _get_or_run("yolov3_weights_to_keras", parameters, git_commit)
        _get_or_run(
            "detector", parameters , git_commit)


if __name__ == '__main__':
    
    if not os.path.exists("models"): os.mkdir("models")
    if not os.path.exists("outputs"): os.mkdir("outputs")
    if sys.argv[4] =="None":
        appendfunc = lambda spp, dataset: "-spp" if spp and dataset =="coco" else "" if dataset == "coco" else "-{}".format(dataset)
        sys.argv[4] = 'models/yolov3{}.h5'.format(appendfunc(int(sys.argv[3]), sys.argv[2]))
        print("model path set to {}".format(sys.argv[4]))
    workflow(darknet_model_path = sys.argv[1], dataset = sys.argv[2], SPP = sys.argv[3], keras_model_path = sys.argv[4], mode = sys.argv[5],\
             size = sys.argv[6], file_name = sys.argv[7], darknet_url = sys.argv[8])
