import json
import re
import argparse

def get_operator_start_time(trace_json, operator_name, tid):
    with open(trace_json, 'r') as f:
        trace_data = json.load(f)

    pattern_name = None
    pattern_tid = None
    if operator_name:
        pattern_name = re.compile(rf"{operator_name}", re.IGNORECASE)
    elif tid:
        pattern_tid = re.compile(rf"tid:{tid}_.+", re.IGNORECASE)
    else:
        raise ValueError("Either operator_name or tid must be provided.")
    operator_name_list = []
    tid_list = []
    ts_list = []
    dur_list = []
    args_list = []
    for operator in trace_data:
        if pattern_name and pattern_name.search(operator.get('name')):
            operator_name_list.append(operator.get('name'))
            tid_list.append(operator.get('tid'))
            ts_list.append(operator.get('ts'))
            dur_list.append(operator.get('dur'))
            args_list.append(operator.get('args'))
        elif pattern_tid and pattern_tid.search(operator.get('tid')):
            operator_name_list.append(operator.get('name'))
            tid_list.append(operator.get('tid'))
            ts_list.append(operator.get('ts'))
            dur_list.append(operator.get('dur'))
            args_list.append(operator.get('args'))

    operator_list = list(zip(operator_name_list, tid_list, ts_list, dur_list, args_list))
    # 按照ts排序
    operator_list.sort(key=lambda x: float(x[2]) if x[2] is not None else float('inf'))

    return operator_list

def get_min_ts(trace_json):
    with open(trace_json, 'r') as f:
        trace_data = json.load(f)
    min_ts = float('inf')
    for operator in trace_data:
        ts = operator.get('ts')
        # print(type(ts))
        ts_float = float(ts) if ts is not None else None
        # print(type(ts_float))
        if ts_float is not None and ts_float < min_ts:
            min_ts = ts_float
    return min_ts

def parse_args():
    parser = argparse.ArgumentParser(description="Extract operator start times from trace JSON")
    parser.add_argument("--file", "-f", type=str, help="Path to the trace JSON file")
    parser.add_argument("--name", "-n", default=None, type=str, help="Name of the operator to extract")
    parser.add_argument("--tid", "-p", default=None, type=int, help="Process ID to filter (optional)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tid = args.tid
    test_trace_json = args.file
    operator_name = args.name
    # operator_name = "hcom_alltoall__612_34_1"
    operator_list = get_operator_start_time(test_trace_json, operator_name, tid)
    min_ts = get_min_ts(test_trace_json)
    print(f"Minimum Timestamp in Trace: {min_ts}")
    if not operator_list:
        print(f"No occurrences of operator '{operator_name}' found.")
    print(f"Operator: {operator_name}")
    # print(operator_list)
    for operator in operator_list:
        print(f"Operator Name: {operator[0]}, Tid: {operator[1]}, Start Time (ms): {(float(operator[2])-min_ts)/1000}, Duration (us): {operator[3]}, Args: {operator[4]}")
    # for i in range(len(operator_list[0])):
    #     print(f"Start Time: {operator_list[0][i]}, Duration: {operator_list[1][i]}, Args: {operator_list[2][i]}")
