import os
import neptune
import contextlib
import hashlib


@contextlib.contextmanager
def open_log(
    project_name,
    exec_name,
    description=None,
    capture_hardware_metrics=False,
    mode=None,
    capture_stdout=False,
    capture_stderr=False,
    flush_period=30,
    level_tag=None,
    sweep_id=None,
    custom_run_id=None
):
    project = (
        os.environ["NEPTUNE_PROJECT"]
        + "/"
        + project_name
    )

    run = neptune.init_run(
        name=custom_run_id,
        project=project,
        description=description,
        capture_hardware_metrics=capture_hardware_metrics,
        mode=mode,
        capture_stdout=capture_stdout,
        capture_stderr=capture_stderr,
        flush_period=flush_period,
        source_files=[exec_name + ".py"] if level_tag == "study-level" else [],
        custom_run_id=hashlib.md5(custom_run_id.encode('utf-8')).hexdigest()[:36]
    )

    run["sys/tags"].add(exec_name)

    if level_tag is not None:
        run["sys/tags"].add(level_tag)

    if sweep_id is not None:
        run["sys/tags"].add(sweep_id)
    
    try:
        custom_id = run["sys/custom_run_id"].fetch()
    except:
        custom_id = "<Unknown>"
    print(f"Run ID: {run._id} / {run._short_id} / {custom_id}")
    print(f"Logging mode:", mode if mode else "async")

    yield run

    run.stop()
