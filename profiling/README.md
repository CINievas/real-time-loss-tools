# Profiling the Real-Time Loss Tools

In order to run `profiling.py`, create a complete runable set of inputs (the same as if you were
about to run a specific calculation), which will be located under a directory of your choice,
e.g. `user/path/inputfiles`. Navigate to `user/path/inputfiles` in your command line (with the
virtual environment activated, assuming you use a virtual environment to run the Real-Time Loss
Tools), and type:

```
user@computer:~/user/path/inputfiles$ python ../../real-time-loss-tools/profiling/profiling.py
```

with as many `../` as necessary to navigate to the `real-time-loss-tools` directory.

The output is stored under:

`user/path/inputfiles/output_profiling/profiling_YYYY_MM_DD_HH_MM_SS.out`

with `YYYY_MM_DD_HH_MM_SS` indicating the date and time in which the profiling was done.

The profiling code also prints to screen:
- the 50 methods with the longest cumulative time results (time spent including time spent in
the functions/methods that it calls)
- the 50 methods with the longest total time results (time spent excluding time spent in the
functions/methods that it calls)
