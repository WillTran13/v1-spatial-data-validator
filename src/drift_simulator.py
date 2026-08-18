from datetime import datetime, timedelta

SENSOR_COUNT = 2
START_TIME = datetime(2026, 1, 1, 0, 0, 0)
TARGET_SENSOR = "sensor_01"

def assign_dimensions(records):
    # sort by frame_id
    sorted_records = sorted(records, key=lambda r: r["frame_id"])

    assigned_records = []

    for index, record in enumerate(sorted_records):
        sensor_id = f"sensor_{index % SENSOR_COUNT:02d}"
        captured_timestamp = START_TIME + timedelta(seconds=index)

        new_record = {
            "frame_id": record["frame_id"],
            "sensor_id": sensor_id,
            "captured_timestamp": captured_timestamp,
        }
        assigned_records.append(new_record)

    return assigned_records

def assign_severity(dimension_rows, target_sensor):
    '''
    add drift severity to each row IN PLACE. modify the input and return it. Only to the sensor ramps; the rest is 0.0
    '''
    targeted_rows = [r for r in dimension_rows if r["sensor_id"] == target_sensor]
    targeted_rows = sorted(targeted_rows, key=lambda r: r["captured_timestamp"])

    n_row = len(targeted_rows)

    for index, row in enumerate(targeted_rows):
        if n_row == 1:
            row["drift_severity"] = 0.0
        else:
            row["drift_severity"] = index / (n_row - 1)

    other_rows = [r for r in dimension_rows if r["sensor_id"] != target_sensor]

    for row in other_rows:
        row["drift_severity"] = 0.0

    return dimension_rows

if __name__ == "__main__":
    fake = [{"frame_id": f"frame_{i}"} for i in range(8)]

    # assign dimension test
    print("Dimension test")
    for row in assign_dimensions(fake):
        print(row)

    # assign severity test
    print("\nSeverity test")
    rows = assign_dimensions(fake)
    for row in assign_severity(rows, TARGET_SENSOR):
        print(row)

    # assign seversity one row test
    print("\nOne row test")
    fake_two = [{"frame_id": f"frame_{i}"} for i in range(2)]
    rows = assign_dimensions(fake_two)
    for row in assign_severity(rows, TARGET_SENSOR):
        print(row)